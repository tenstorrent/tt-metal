# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure served OCR latency and decode throughput per vision bucket.

Streams the response so the two halves can be separated the way a user
experiences them:

*TTFT* is the wall time from sending the request to the first content token. For
this model it covers the host patch embedding, the vision tower over the whole
padded bucket, the splice, and the text prefill. It grows with image size
because the tower's work does, which is why results are reported per bucket
rather than as one number.

*Decode rate* is the remaining tokens over the remaining time, so it excludes
prefill and is comparable to a text model's tokens/s/user.

Each bucket is warmed before timing, then measured several times and reported by
median: the first request after a period of idleness is not representative, and
a single sample on a device shared with nothing else still moves by tens of
milliseconds.

Run (server on :8100)::

    python models/demos/blackhole/paddleocr_vl/tests/serve_perf.py --reps 5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import time
import urllib.request

from PIL import Image

DEMO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
GOLDEN = os.path.join(DEMO, "golden", "hf_goldens.json")
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PROMPT = "OCR:"

# Gates from the bring-up plan. TTFT is quoted for the largest bucket, which is
# the worst case and the one a full page hits.
TTFT_GATE_S = 0.600
DECODE_GATE_TOKS = 150.0


def data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def stream_once(base: str, payload_url: str, max_tokens: int, timeout: float):
    """Return (ttft_s, decode_toks_per_s, n_tokens, total_s)."""
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": payload_url}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    total = time.perf_counter() - t0

    if ttft is None or n <= 1:
        return ttft, None, n, total
    return ttft, (n - 1) / (total - ttft), n, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--samples", default="sign_exit,receipt_cafe,table_quarterly,page_table_large")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"

    with open(GOLDEN) as f:
        goldens = {s["name"]: s for s in json.load(f)["samples"]}

    names = [n for n in a.samples.split(",") if n in goldens]
    rows = []

    for name in names:
        g = goldens[name]
        url = data_url(os.path.join(DEMO, g["image"]))
        n_img_tok = g.get("n_image_tokens")

        # Warm this bucket's path before timing it.
        stream_once(base, url, a.max_tokens, a.timeout)

        ttfts, rates, counts = [], [], []
        for _ in range(a.reps):
            ttft, rate, n, _total = stream_once(base, url, a.max_tokens, a.timeout)
            if ttft is None:
                print(f"  {name}: no content returned")
                break
            ttfts.append(ttft)
            counts.append(n)
            if rate is not None:
                rates.append(rate)

        if not ttfts:
            continue
        rows.append(
            {
                "name": name,
                "bucket": g["bucket"],
                "img_tokens": n_img_tok,
                "gen": int(statistics.median(counts)),
                "ttft": statistics.median(ttfts),
                "ttft_min": min(ttfts),
                "decode": statistics.median(rates) if rates else float("nan"),
            }
        )
        print(
            f"  {name:20s} bucket={g['bucket']:5d} imgtok={n_img_tok or 0:5d} "
            f"ttft={rows[-1]['ttft'] * 1000:7.1f}ms decode={rows[-1]['decode']:6.1f} tok/s gen={rows[-1]['gen']}"
        )

    print("\n=================== S6 PERF ===================")
    print(f"{'sample':20s} {'bkt':>5s} {'imgtok':>6s} {'gen':>4s} {'TTFT':>9s} {'TTFTmin':>9s} {'decode':>11s}")
    for r in rows:
        print(
            f"{r['name']:20s} {r['bucket']:5d} {r['img_tokens'] or 0:6d} {r['gen']:4d} "
            f"{r['ttft'] * 1000:8.1f}ms {r['ttft_min'] * 1000:8.1f}ms {r['decode']:8.1f} t/s"
        )

    if not rows:
        print("===============================================")
        return 1

    worst = max(rows, key=lambda r: r["bucket"])
    decodes = [r["decode"] for r in rows if r["decode"] == r["decode"]]
    med_decode = statistics.median(decodes) if decodes else float("nan")
    ttft_pass = worst["ttft"] <= TTFT_GATE_S
    decode_pass = med_decode >= DECODE_GATE_TOKS
    print(
        f"\ngates: TTFT <= {TTFT_GATE_S * 1000:.0f}ms on the largest bucket, "
        f"decode >= {DECODE_GATE_TOKS:.0f} tok/s/user"
    )
    print(
        f"  largest bucket ({worst['bucket']}): TTFT {worst['ttft'] * 1000:.1f}ms -> {'PASS' if ttft_pass else 'MISS'}"
    )
    print(f"  median decode: {med_decode:.1f} tok/s -> {'PASS' if decode_pass else 'MISS'}")
    print("===============================================")
    return 0 if (ttft_pass and decode_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
