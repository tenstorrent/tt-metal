#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device stress client: a prefill batch that mixes the FIRST chunk of an image prompt (image first, position 0) with continuation chunks of long text prompts, repeated.
usage: serving_mixed_image_batch.py [--iters 30] [--long 40000] [--out res.json]
Per iteration: 2 fresh long text prompts (random ids, ~--long tokens) start together; image prompts (image first + ~3k tokens of text, fresh image each time) are sent at +3 s, +7 s and +11 s
so that their first chunk lands in a step that also runs the long prompts' continuation chunks. Fails (exit 1) on a request error, a wrong image answer or a stall > 300 s (hang)."""
import base64
import io
import json
import random
import sys
import threading
import time
import urllib.request

ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 30
LONG = int(sys.argv[sys.argv.index("--long") + 1]) if "--long" in sys.argv else 40000
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
VOCAB = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()


def post(path, body):
    r = urllib.request.Request(
        "http://127.0.0.1:8000" + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(r, timeout=300))


def img(num):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (448, 224), (255, 255, 255))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
    except Exception:
        f = ImageFont.load_default()
    d.text((20, 50), str(num), fill=(200, 0, 0), font=f)
    b = io.BytesIO()
    im.save(b, "PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()},
    }


rows, bad = [], []


def long_req(seed, k):
    r = random.Random(seed)
    ids = [r.randint(1000, 100000) for _ in range(LONG)]
    t0 = time.time()
    try:
        post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 8, "temperature": 0})
        rows.append({"kind": "long", "it": k, "s": round(time.time() - t0, 1)})
    except Exception as e:
        bad.append({"kind": "long", "it": k, "err": repr(e)})


def img_req(k, delay, j):
    time.sleep(delay)
    num = str(random.Random(k * 10 + j).randrange(1000, 9999))
    r = random.Random(k * 7 + j)
    txt = (
        " ".join(r.choice(VOCAB) + str(r.randrange(1000)) for _ in range(1500))
        + "\nWhat number is written in the image? Answer with the number only."
    )
    t0 = time.time()
    try:
        d = post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": [img(num), {"type": "text", "text": txt}]}],
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        a = d["choices"][0]["message"]["content"]
        row = {"kind": "image", "it": k, "j": j, "ok": num in a, "answer": a[:30], "s": round(time.time() - t0, 1)}
        rows.append(row)
        if not row["ok"]:
            bad.append(row)
    except Exception as e:
        bad.append({"kind": "image", "it": k, "j": j, "err": repr(e)})


for k in range(ITERS):
    ts = [
        threading.Thread(target=long_req, args=(1000 * k + i + int(time.time()) % 1000 * 100000, k)) for i in range(2)
    ] + [threading.Thread(target=img_req, args=(k, d, j)) for j, d in enumerate((3, 7, 11))]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(f"iter {k}: {time.time() - t0:.1f} s, bad so far {len(bad)}", flush=True)
    if bad:
        print("FIRST BAD", bad[0], flush=True)
        break
print("RESULT", "PASS" if not bad else "FAIL", len(rows), "ok requests", flush=True)
if OUT:
    json.dump({"rows": rows, "bad": bad}, open(OUT, "w"), indent=1)
sys.exit(0 if not bad else 1)
