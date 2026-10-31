#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prefill positions computed per prompt token for many-image prompts (the image-cut recompute waste), from the server log: usage: serving_image_recompute_waste.py --log server.log [--out r.json]
Per prompt of N images (cold, then a repeat): sum of the lengths of the logged prefill pieces ``chunks [(a, b)]`` (positions computed, restores not counted) / usage.prompt_tokens."""
import base64
import io
import json
import random
import re
import sys
import time
import urllib.request

A = sys.argv
LOG = A[A.index("--log") + 1]
OUT = A[A.index("--out") + 1] if "--out" in A else None
M = "deepseek-ai/DeepSeek-V4.1-Flash"
off = [0]


def png(i):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (112, 112), (255, 255, 255))
    ImageDraw.Draw(im).text((10, 40), f"N{i}", fill=(200, 0, 0))
    b = io.BytesIO()
    im.save(b, "PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()},
    }


def log_delta():
    with open(LOG, errors="replace") as f:
        f.seek(off[0])
        t = f.read()
        off[0] = f.tell()
    comp = 0
    for m in re.finditer(r"chunks \[(.*?)\] in slots", t):
        comp += sum(int(b) - int(a) for a, b in re.findall(r"\((\d+), (\d+)\)", m.group(1)))
    rest = [int(x) for x in re.findall(r"restored at (\d+) tokens", t)]
    return comp, rest, len(re.findall(r"recomputes from 0|without carried state", t))


def ask(content):
    body = {"model": M, "messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0}
    r = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=3600))
    return d["usage"]["prompt_tokens"], round(time.time() - t0, 1), d["choices"][0]["message"]["content"][:30]


log_delta()
res = []
run = random.Random(int(time.time())).randrange(10**6)
for n in (10, 50, 100, 250):
    content = (
        [{"type": "text", "text": f"run {run}-{n}: "}]
        + [png(run * 1000 + i) for i in range(n)]
        + [{"type": "text", "text": "How many images are there? Number only."}]
    )
    for lab in ("cold", "repeat"):
        pt, dt, ans = ask(content)
        comp, rest, rec = log_delta()
        row = {
            "images": n,
            "run": lab,
            "prompt_tokens": pt,
            "computed_positions": comp,
            "ratio": round(comp / pt, 3),
            "restores": rest,
            "recompute_msgs": rec,
            "s": dt,
            "answer": ans,
        }
        res.append(row)
        print(row, flush=True)
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
