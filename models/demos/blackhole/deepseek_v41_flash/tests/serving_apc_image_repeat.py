#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device regression test (needs a running DSV4.1 server with image input and prefix caching on, http://127.0.0.1:8000): repeated image prompts must answer
correctly on every repeat. Before the fix an image-first prompt of >= ~640 tokens answered a wrong number on EVERY prefix-cache hit (vLLM reported the
cached range as computed, so the model got the placeholder tokens without the pixels, and recomputed them from position 0 without the image).

usage: serving_apc_image_repeat.py [--out res.json] [--port 8000]       exit code 0 = every case passes
Standard library + PIL only (token counts are approximated by 2 tokens per filler word; the server's usage.prompt_tokens is reported).
"""
import base64
import io
import json
import random
import sys
import time
import urllib.request

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
VOCAB = "alpha beta gamma delta river stone cloud forest engine window garden bridge market lantern copper meadow harbor violet anchor pencil".split()


def words(seed, n_tokens):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) + str(r.randrange(1000)) for _ in range(n_tokens // 2))


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


def ask(content, salt=None, mt=16, messages=None):
    body = {
        "model": MODEL,
        "messages": messages or [{"role": "user", "content": content}],
        "max_tokens": mt,
        "temperature": 0,
    }
    if salt is not None:
        body["cache_salt"] = salt
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"], round(time.time() - t0, 2)


def cached_total():
    try:
        t = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    except Exception:
        return None
    return sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith("vllm:prompt_tokens_cached_total"))


Q = "What number is written in the image? Answer with the number only."
res, fails = [], []


def case(name, content, expect, salt, repeats, mt=16):
    for k in range(repeats):
        c0 = cached_total()
        a, n, dt = ask(content, salt, mt)
        c1 = cached_total()
        ok = expect in a
        row = {
            "case": name,
            "run": k,
            "answer": a[:40],
            "expect": expect,
            "ok": ok,
            "prompt_tokens": n,
            "cached": None if c0 is None else c1 - c0,
            "s": dt,
        }
        res.append(row)
        print(row, flush=True)
        if not ok:
            fails.append(row)


run_id = str(int(time.time()))
for pad in (700, 1200, 6000):  # image first, then text; the 1st run is cold, every repeat is a prefix-cache hit
    num = str(random.Random(pad).randrange(1000, 9999))
    case(
        f"image_first_{pad}",
        [img(num), {"type": "text", "text": words(pad, pad) + "\n" + Q}],
        num,
        f"if{pad}-{run_id}",
        4,
    )
num = str(random.Random(1).randrange(1000, 9999))  # image last
case(
    "image_last_1200",
    [{"type": "text", "text": words(77, 1200) + "\nThe image follows. " + Q}, img(num)],
    num,
    "il-" + run_id,
    3,
)
num = str(random.Random(2).randrange(1000, 9999))  # image in the middle
case(
    "image_middle",
    [{"type": "text", "text": words(78, 1200)}, img(num), {"type": "text", "text": words(79, 600) + "\n" + Q}],
    num,
    "im-" + run_id,
    3,
)
# two different images with the same text prefix must never share a cache entry
n1, n2 = "4821", "7305"
txt = {"type": "text", "text": words(80, 1500) + "\n" + Q}
for lab, num in (("A1", n1), ("B1", n2), ("A2", n1), ("B2", n2), ("A3", n1)):
    case(f"two_images_same_text_{lab}", [img(num), txt], num, "two-" + run_id, 1)
# same image, different text after the image and the same long text: answers follow the question
num = "6139"
head = words(81, 1500)
case("same_image_q_number", [img(num), {"type": "text", "text": head + "\n" + Q}], num, "sq-" + run_id, 2)
for k in range(2):
    a, n, dt = ask(
        [img(num), {"type": "text", "text": head + "\nWhat colour are the digits in the image? Answer with one word."}],
        "sq-" + run_id,
    )
    row = {"case": "same_image_q_colour", "run": k, "answer": a[:40], "ok": "red" in a.lower(), "prompt_tokens": n}
    res.append(row)
    print(row, flush=True)
    if not row["ok"]:
        fails.append(row)
# cache_salt variants of one prompt: none, salt A, salt B, A again
num = "2958"
content = [img(num), {"type": "text", "text": words(82, 1500) + "\n" + Q}]
for lab, salt in (
    ("none", None),
    ("A", "sa" + run_id),
    ("B", "sb" + run_id),
    ("A_again", "sa" + run_id),
    ("none_again", None),
):
    case(f"salt_{lab}", content, num, salt, 1)
# text-only prefix caching must be untouched: the repeat hits (cached tokens > 0) and answers the same
code = "5516"
tmsg = (
    words(83, 8000)
    + f"\nThe secret code is {code}. Remember it.\n"
    + words(84, 200)
    + "\nWhat is the secret code? Answer with the number only."
)
c0 = cached_total()
a1, n, t1 = ask(tmsg, "txt-" + run_id)
c1 = cached_total()
a2, n, t2 = ask(tmsg, "txt-" + run_id)
c2 = cached_total()
row = {
    "case": "text_only_repeat",
    "answers": [a1, a2],
    "ok": code in a1 and code in a2,
    "cached_second": None if c1 is None else c2 - c1,
    "ttft_cold_hit": [t1, t2],
}
if c1 is not None and c2 - c1 <= 0:
    row["ok"] = False
res.append(row)
print(row, flush=True)
if not row["ok"]:
    fails.append(row)
print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)} of {len(res)} runs)", flush=True)
if OUT:
    json.dump({"rows": res, "fails": fails}, open(OUT, "w"), indent=1)
sys.exit(0 if not fails else 1)
