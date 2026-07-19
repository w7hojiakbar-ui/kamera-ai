"""
Kamera AI Kuzatuv Tizimi - Prototip
------------------------------------
Telefon (yoki istalgan kamera) rasmni shu serverga yuboradi.
Server harakatni tekshiradi, agar harakat bo'lsa GPT-4o-mini'ga yuborib
tavsif oladi va SQLite bazaga vaqt bilan yozadi.

So'rov qilish uchun: GET /ask?vaqt=14:30  yoki  /ask?savol=oxirgi 10 daqiqada nima bo'ldi

Ishga tushirish (lokal):
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import io
import sqlite3
import base64
import time
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageChops
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from openai import OpenAI

# ---------- Sozlamalar ----------
DB_PATH = os.environ.get("DB_PATH", "kuzatuv.db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MOTION_THRESHOLD = float(os.environ.get("MOTION_THRESHOLD", "12.0"))  # necha % o'zgarsa "harakat" deb hisoblansin
MIN_SECONDS_BETWEEN_CALLS = int(os.environ.get("MIN_SECONDS_BETWEEN_CALLS", "8"))  # bitta kamera uchun AI'ga eng kamida necha soniyada bir marta murojaat qilish mumkin (token tejash)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Kamera AI Kuzatuv")

# Har bir kamera uchun oxirgi kadr va oxirgi AI chaqiruv vaqtini xotirada saqlaymiz
_last_frames = {}   # {camera_id: PIL.Image}
_last_call_time = {}  # {camera_id: timestamp}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS voqealar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            vaqt TEXT,
            tavsif TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def motion_percent(prev_img: Image.Image, curr_img: Image.Image) -> float:
    """Ikki kadr orasidagi farqni % da qaytaradi (oddiy pixel-diff usuli)."""
    prev_small = prev_img.convert("L").resize((160, 120))
    curr_small = curr_img.convert("L").resize((160, 120))
    diff = ImageChops.difference(prev_small, curr_small)
    arr = np.array(diff)
    changed_pixels = np.sum(arr > 25)  # 25 dan katta farq - "o'zgargan" piksel
    total_pixels = arr.size
    return (changed_pixels / total_pixels) * 100


def describe_frame_with_ai(image_bytes: bytes) -> str:
    """Rasmni GPT-4o-mini'ga yuborib, qisqa tavsif oladi."""
    if client is None:
        return "[XATO: OPENAI_API_KEY o'rnatilmagan]"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen zavod/xona kuzatuv kamerasi uchun ishlovchi AI qorovulsan. "
                    "Rasmda nima ko'rinayotganini 1-2 gapda, aniq va qisqa, o'zbek tilida yoz. "
                    "Odamlar, ularning harakati, buyumlar, potentsial xavf yoki g'ayrioddiy holatlarga e'tibor ber. "
                    "Agar hech narsa bo'lmasa, shunchaki 'Xonada o'zgarish yo'q, hammasi tinch' deb yoz."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bu kadrda nima sodir bo'layapti?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            },
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


def save_event(camera_id: str, tavsif: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO voqealar (camera_id, vaqt, tavsif) VALUES (?, ?, ?)",
        (camera_id, datetime.now().isoformat(timespec="seconds"), tavsif),
    )
    conn.commit()
    conn.close()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Kamera AI Kuzatuv Tizimi ishlayapti ✅</h2>
    <p>Telefon kamerasini ulash uchun: <a href="/kamera">/kamera</a> sahifasini oching</p>
    <p>Rasm yuborish: POST /frame (multipart form: camera_id, file)</p>
    <p>So'rov: GET /ask?savol=oxirgi 1 soatda nima bo'ldi</p>
    <p>Barcha voqealar: GET /voqealar</p>
    """


@app.get("/kamera", response_class=HTMLResponse)
def kamera_page():
    with open(os.path.join(os.path.dirname(__file__), "telefon_kamera.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/frame")
async def upload_frame(camera_id: str = Form(...), file: UploadFile = File(...)):
    """Telefon/kamera bu endpoint'ga har necha soniyada bir rasm yuboradi."""
    image_bytes = await file.read()
    curr_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    now = time.time()
    prev_img = _last_frames.get(camera_id)
    last_call = _last_call_time.get(camera_id, 0)

    result = {"camera_id": camera_id, "harakat_foizi": 0.0, "ai_chaqirildi": False}

    if prev_img is not None:
        pct = motion_percent(prev_img, curr_img)
        result["harakat_foizi"] = round(pct, 2)

        enough_time_passed = (now - last_call) >= MIN_SECONDS_BETWEEN_CALLS

        if pct >= MOTION_THRESHOLD and enough_time_passed:
            tavsif = describe_frame_with_ai(image_bytes)
            save_event(camera_id, tavsif)
            _last_call_time[camera_id] = now
            result["ai_chaqirildi"] = True
            result["tavsif"] = tavsif
    else:
        # Birinchi kadr - hali solishtirish uchun oldingi kadr yo'q, faqat saqlaymiz
        result["izoh"] = "Birinchi kadr qabul qilindi, keyingi kadrlar bilan solishtiriladi"

    _last_frames[camera_id] = curr_img
    return JSONResponse(result)


@app.get("/voqealar")
def list_events(camera_id: str = None, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if camera_id:
        rows = conn.execute(
            "SELECT * FROM voqealar WHERE camera_id=? ORDER BY id DESC LIMIT ?",
            (camera_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM voqealar ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/ask")
def ask(savol: str = Query(..., description="Masalan: 'oxirgi 1 soatda nima bo'ldi'"),
         soatlar: int = Query(2, description="Nechta soatlik tarixni qidirish kerak")):
    """Foydalanuvchi savol beradi, biz mos vaqtdagi voqealarni bazadan olib,
    GPT-4o-mini'ga context sifatida berib, tabiiy tilda javob qildiramiz."""
    if client is None:
        return {"javob": "[XATO: OPENAI_API_KEY o'rnatilmagan]"}

    since = (datetime.now() - timedelta(hours=soatlar)).isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM voqealar WHERE vaqt >= ? ORDER BY vaqt ASC", (since,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"javob": f"Oxirgi {soatlar} soat ichida hech qanday qayd etilgan voqea topilmadi."}

    context_lines = [f"{r['vaqt']} ({r['camera_id']}): {r['tavsif']}" for r in rows]
    context_text = "\n".join(context_lines)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen zavod kuzatuv tizimining AI yordamchisisan. "
                    "Quyida vaqt bo'yicha tartiblangan kamera voqealari ro'yxati berilgan. "
                    "Foydalanuvchi savoliga shu ma'lumotlar asosida, o'zbek tilida, aniq va qisqa javob ber. "
                    "Agar so'ralgan vaqtda ma'lumot bo'lmasa, shunday deb ayt."
                ),
            },
            {
                "role": "user",
                "content": f"Voqealar ro'yxati:\n{context_text}\n\nSavol: {savol}",
            },
        ],
        max_tokens=300,
    )
    return {"javob": response.choices[0].message.content.strip(), "topilgan_voqealar_soni": len(rows)}
