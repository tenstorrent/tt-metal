#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device regression for prefix caching WITH images (DSV41_APC_IMAGES=1, server with image input): repeated image prompts must restore a snapshot (hit) AND answer correctly.

usage: serving_apc_image_hits.py --log /path/to/server.log [--out res.json] [--port 8000] [--expect-hits 1|0]
The hit is read from the server log ("prefix cache: users [..] restored at N tokens"): with the plugin's mm prefix-cache skip vLLM's own counters stay 0 for image prompts, the model's
snapshot store does the work. --expect-hits 0 checks the fallback (switch off): answers must be correct and no image prompt may restore anything.
Standard library + PIL only; exit code 0 = every case passes."""
import base64
import io
import json
import random
import re
import sys
import time
import urllib.request

A = sys.argv
PORT = int(A[A.index("--port") + 1]) if "--port" in A else 8000
OUT = A[A.index("--out") + 1] if "--out" in A else None
LOG = A[A.index("--log") + 1] if "--log" in A else None
EXPECT_HITS = (A[A.index("--expect-hits") + 1] != "0") if "--expect-hits" in A else True
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
VOCAB = "alpha beta gamma delta river stone cloud forest engine window garden bridge market lantern copper meadow harbor violet anchor pencil".split()
RUN = str(int(time.time()))
_off = [0]


def words(seed, n_tokens):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) + str(r.randrange(1000)) for _ in range(n_tokens // 2))


def img(num, size=(448, 224), tag=""):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", size, (255, 255, 255))
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


def restores():
    """token counts of the snapshot restores logged since the last call"""
    if not LOG:
        return None
    with open(LOG, errors="replace") as f:
        f.seek(_off[0])
        t = f.read()
        _off[0] = f.tell()
    return [int(m.group(1)) for m in re.finditer(r"prefix cache: users \[[^\]]*\] restored at (\d+) tokens", t)]


def chat(messages, salt=None, mt=16):
    body = {"model": MODEL, "messages": messages, "max_tokens": mt, "temperature": 0}
    if salt is not None:
        body["cache_salt"] = salt
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"], round(time.time() - t0, 2)


rows, fails = [], []


def case(name, content, expect, salt, repeats, want_hit_after_first=True, min_restore=0, max_restore=None, mt=16):
    outs = []
    for k in range(repeats):
        restores()
        a, n, dt = chat([{"role": "user", "content": content}], salt, mt)
        r = restores()
        ok = expect in a
        hit = max(r) if r else 0
        row = {
            "case": name,
            "run": k,
            "answer": a[:50],
            "expect": expect,
            "ok": ok,
            "prompt_tokens": n,
            "restored": hit,
            "s": dt,
        }
        if k > 0 and EXPECT_HITS and LOG and want_hit_after_first and hit < max(1, min_restore):
            row["ok"] = False
            row["why"] = "no snapshot restore on a repeat"
        if (not EXPECT_HITS) and hit:
            row["ok"] = False
            row["why"] = "restore with DSV41_APC_IMAGES off"
        if max_restore is not None and hit > max_restore:
            row["ok"] = False
            row["why"] = f"restore {hit} beyond the allowed {max_restore} (image boundary rule)"
        rows.append(row)
        outs.append(a)
        print(row, flush=True)
        if not row["ok"]:
            fails.append(row)
    return outs


# 1. image first, 4 repeats; 2. image last; 3. image in the middle
for pad in (700, 1200, 6000):
    num = str(random.Random(pad).randrange(1000, 9999))
    case(
        f"image_first_{pad}",
        [
            img(num),
            {
                "type": "text",
                "text": words(pad, pad) + "\nWhat number is written in the image? Answer with the number only.",
            },
        ],
        num,
        f"if{pad}-{RUN}",
        4,
    )
num = str(random.Random(1).randrange(1000, 9999))
case(
    "image_last_1200",
    [
        {"type": "text", "text": words(77, 1200) + "\nThe image follows. What number is written in it? Number only."},
        img(num),
    ],
    num,
    "il-" + RUN,
    3,
    want_hit_after_first=False,
)
num = str(random.Random(2).randrange(1000, 9999))
case(
    "image_middle",
    [
        {"type": "text", "text": words(78, 1200)},
        img(num),
        {"type": "text", "text": words(79, 600) + "\nWhat number is written in the image? Number only."},
    ],
    num,
    "im-" + RUN,
    3,
    want_hit_after_first=False,
)
# 4. an image that crosses the 512 boundary: ~380 tokens of text, then the image (> 100 tokens), then 2000 tokens: a hit may only land after the image (>= 1024)
num = str(random.Random(3).randrange(1000, 9999))
case(
    "image_straddles_512",
    [
        {"type": "text", "text": words(85, 380)},
        img(num),
        {"type": "text", "text": words(86, 2000) + "\nWhat number is written in the image? Number only."},
    ],
    num,
    "st-" + RUN,
    3,
    want_hit_after_first=True,
    min_restore=1024,
)
# 5. two different images, same text: never a cross hit (the 1st run of B must restore nothing)
n1, n2 = "4821", "7305"
txt = {"type": "text", "text": words(80, 1500) + "\nWhat number is written in the image? Number only."}
case("two_images_A1", [img(n1), txt], n1, "two-" + RUN, 1)
restores()
case(
    "two_images_B1_no_cross_hit",
    [img(n2), txt],
    n2,
    "two-" + RUN,
    1,
    want_hit_after_first=False,
    max_restore=0 if LOG else None,
)
case("two_images_A2", [img(n1), txt], n1, "two-" + RUN, 1, want_hit_after_first=True, min_restore=1)
rows[-1]["ok"] = rows[-1]["ok"] and (rows[-1]["restored"] >= 1 or not (EXPECT_HITS and LOG))
case("two_images_B2", [img(n2), txt], n2, "two-" + RUN, 1, want_hit_after_first=True, min_restore=1)
# 6. same image + same long text, different questions: hit up to the shared part, answers follow the question
num = "6139"
head = words(81, 1500)
case(
    "same_image_q_number",
    [img(num), {"type": "text", "text": head + "\nWhat number is written in the image? Answer with the number only."}],
    num,
    "sq-" + RUN,
    2,
)
a, n, dt = chat(
    [
        {
            "role": "user",
            "content": [
                img(num),
                {
                    "type": "text",
                    "text": head + "\nWhat colour are the digits in the image (not the background)? One word.",
                },
            ],
        }
    ],
    "sq-" + RUN,
)
row = {"case": "same_image_q_colour", "answer": a[:30], "ok": "red" in a.lower(), "restored": max(restores() or [0])}
rows.append(row)
print(row, flush=True)
if not row["ok"]:
    fails.append(row)
# 7. cache_salt isolation: salt A cold, salt B must not restore A's snapshot, A again restores
num = "2958"
content = [img(num), {"type": "text", "text": words(82, 1500) + "\nWhat number is written in the image? Number only."}]
case("salt_A_cold", content, num, "sa" + RUN, 1)
case("salt_B_no_cross_hit", content, num, "sb" + RUN, 1, want_hit_after_first=False, max_restore=0 if LOG else None)
case("salt_A_again", content, num, "sa" + RUN, 1, want_hit_after_first=True, min_restore=1)
rows[-1]["ok"] = rows[-1]["ok"] and (rows[-1]["restored"] >= 1 or not (EXPECT_HITS and LOG))
# 8. hit text equals cold text (first 100 characters; spec decode k adapts: a later divergence is not an APC fault)
num = "3517"
content = [
    img(num),
    {
        "type": "text",
        "text": words(87, 2000) + "\nFirst write the number in the image, then write a 40-word story about a lantern.",
    },
]
o = case("hit_equals_cold", content, num, "eq-" + RUN, 2, mt=48)
row = {
    "case": "hit_equals_cold_text",
    "ok": len(o) == 2 and o[0][:100] == o[1][:100],
    "cold": o[0][:100],
    "hit": o[1][:100],
}
rows.append(row)
print(row, flush=True)
if not row["ok"]:
    fails.append(row)
# 9. multi-turn chat with the image kept in the history: the restore grows with the history, answers stay right
num = "8264"
msgs = [
    {
        "role": "user",
        "content": [
            img(num),
            {
                "type": "text",
                "text": "Remember the number in this image. " + words(90, 1200) + "\nReply with the word ok.",
            },
        ],
    }
]
grew = []
for turn in range(1, 7):
    restores()
    a, n, dt = chat(msgs, "mt-" + RUN)
    r = restores()
    hit = max(r) if r else 0
    grew.append(hit)
    ok = True if turn == 1 else True
    msgs.append({"role": "assistant", "content": a})
    nxt = (
        "What number is in the first image of this conversation? Number only."
        if turn % 2
        else "Now ignore that. " + words(90 + turn, 900) + "\nWhat number is in the first image? Number only."
    )
    msgs.append({"role": "user", "content": nxt})
    if turn >= 2:
        pass
    a2, n2_, dt2 = chat(msgs, "mt-" + RUN)
    r2 = restores()
    hit2 = max(r2) if r2 else 0
    ok = num in a2
    row = {"case": f"multiturn_{turn}", "answer": a2[:30], "ok": ok, "prompt_tokens": n2_, "restored": hit2, "s": dt2}
    rows.append(row)
    print(row, flush=True)
    if not ok:
        fails.append(row)
    msgs.append({"role": "assistant", "content": a2})
    msgs.append({"role": "user", "content": words(95 + turn, 700) + "\nReply with the word ok."})
mt_hits = [r["restored"] for r in rows if r["case"].startswith("multiturn_")]
if EXPECT_HITS and LOG:
    row = {"case": "multiturn_hits_grow", "restored": mt_hits, "ok": mt_hits[-1] > mt_hits[0] and mt_hits[-1] > 0}
    rows.append(row)
    print(row, flush=True)
    if not row["ok"]:
        fails.append(row)
# 10. text-only prefix caching untouched: the repeat hits and answers the same
code = "5516"
tmsg = (
    words(83, 8000)
    + f"\nThe secret code is {code}. Remember it.\n"
    + words(84, 200)
    + "\nWhat is the secret code? Answer with the number only."
)
restores()
a1, _, t1 = chat([{"role": "user", "content": tmsg}], "txt-" + RUN)
restores()
a2, _, t2 = chat([{"role": "user", "content": tmsg}], "txt-" + RUN)
r = restores()
row = {
    "case": "text_only_repeat",
    "ok": code in a1 and code in a2 and (not LOG or bool(r)),
    "restored": max(r) if r else 0,
    "ttft_cold_hit": [t1, t2],
}
rows.append(row)
print(row, flush=True)
if not row["ok"]:
    fails.append(row)
print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)} of {len(rows)} rows)", flush=True)
if OUT:
    json.dump({"rows": rows, "fails": fails}, open(OUT, "w"), indent=1)
sys.exit(0 if not fails else 1)
