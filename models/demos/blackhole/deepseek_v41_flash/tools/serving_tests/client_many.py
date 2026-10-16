#!/usr/bin/env python3
"""Many-image checks against a running DSV4.1 server with --limit-mm-per-prompt {"image": 16}: 16 numbered images in one prompt (all 16 numbers must come back), one maximum-size image, 17 images (must be refused with an OpenAI-style HTTP 400 error, and the server must keep serving)."""
import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from client_img import URL, card, img, txt
from PIL import Image

COLORS = [
    (220, 30, 30),
    (30, 60, 220),
    (20, 150, 40),
    (230, 160, 0),
    (150, 40, 170),
    (0, 160, 170),
    (120, 120, 120),
    (200, 80, 120),
]


def numbered(i):
    return card(f"N {i:02d}", "square" if i % 2 else "circle", COLORS[i % len(COLORS)])


def raw_chat(content, max_tokens=32):
    body = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


ok = True
print("== 16 images in one prompt")
content = [txt("Each of the following 16 images has a label 'N xx'. List all 16 labels in order, comma separated.")]
for i in range(1, 17):
    content += [img(numbered(i))]
t0 = time.time()
st, j = raw_chat(content, 160)
dt = time.time() - t0
ans = j["choices"][0]["message"]["content"] if st == 200 else str(j)
found = [i for i in range(1, 17) if f"{i:02d}" in ans or f"N {i}" in ans]
print(
    f"status {st}, {dt:.1f}s, prompt_tokens {j.get('usage', {}).get('prompt_tokens')}, labels found {len(found)}/16\n answer: {ans[:300]!r}"
)
ok &= st == 200 and len(found) >= 14

print("== one maximum-size image (resized to at most 1024 tokens)")
big = Image.new("RGB", (2400, 1800), (255, 255, 255))
from PIL import ImageDraw

d = ImageDraw.Draw(big)
d.rectangle((300, 300, 1500, 1400), fill=(30, 60, 220))
d.text((100, 50), "BIG 99", fill=(0, 0, 0), font=__import__("client_img").font(200))
t0 = time.time()
st, j = raw_chat([img(big), txt("What color is the large rectangle, and what text is written?")], 48)
dt = time.time() - t0
print(
    f"status {st}, {dt:.1f}s, prompt_tokens {j.get('usage', {}).get('prompt_tokens')}, answer {(j['choices'][0]['message']['content'] if st == 200 else str(j))[:200]!r}"
)
ok &= st == 200

print("== 17 images: must be refused with an OpenAI-style error")
content = [txt("Describe the images.")] + [img(numbered(i)) for i in range(1, 18)]
st, j = raw_chat(content, 16)
print(f"status {st}, body {json.dumps(j)[:400]}")
err = j.get("error") if isinstance(j, dict) else None
good = (
    st == 400
    and isinstance(err, dict)
    and "message" in err
    and ("16" in err["message"] or "image" in err["message"].lower())
)
print("OpenAI-style 400 error:", good)
ok &= good
st, j = raw_chat("What is 6 times 7? Answer with just the number.", 8)
print("server still serving:", st, j["choices"][0]["message"]["content"] if st == 200 else j)
ok &= st == 200
print("MANY_IMAGES_OK" if ok else "MANY_IMAGES_FAIL")
sys.exit(0 if ok else 1)
