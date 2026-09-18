# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hammer a running PaddleOCR-VL server to shake out trace and compile faults.

The vision tower compiles a separate program set for each padded patch count,
and the decoder captures a decode trace. That combination has a specific,
well-documented failure mode on tt-metal: a compile issued while a trace is
parked can corrupt the trace (#48536). The symptom is not an exception on the
compiling request -- that one usually succeeds -- but a *later* request that
hangs or returns garbage. A single pass in ascending bucket order, which is what
``serve_smoke.py`` does, is exactly the sequence least likely to expose it.

So this drives the server three ways:

*Order.* Buckets are visited in a seeded random order rather than ascending, so
a fresh server compiles a large bucket while a small one's trace is already
parked.

*Repetition.* Every image is requested repeatedly across passes, and each
response is compared against that image's first response. Drift between passes
is the signature of a clobbered trace; a hang shows up as a request timeout.

*Depth.* A long run of back-to-back requests on one image checks that nothing
degrades with steady-state use.

Determinism is the assertion rather than accuracy: temperature is 0, so the same
image must produce byte-identical text every time. Accuracy against the
HuggingFace goldens is ``serve_smoke.py``'s job.

Run (server on :8100)::

    python models/demos/blackhole/paddleocr_vl/tests/serve_hardening.py --passes 3 --sequential 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.request

from PIL import Image

DEMO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
GOLDEN = os.path.join(DEMO, "golden", "hf_goldens.json")
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PROMPT = "OCR:"

# One image per vision bucket. Kept small on purpose: the point is bucket
# coverage and repetition, not corpus breadth.
DEFAULT_SAMPLES = "sign_exit,receipt_cafe,table_quarterly,page_manual"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ocr_request(base: str, payload_url: str, max_tokens: int, timeout: float):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
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
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body["choices"][0]["message"]["content"], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--passes", type=int, default=3, help="times to cycle the bucket set")
    ap.add_argument("--sequential", type=int, default=50, help="back-to-back requests on one image")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout; a hang trips this")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--samples", default=DEFAULT_SAMPLES)
    a = ap.parse_args()

    base = f"http://{a.host}:{a.port}"
    rng = random.Random(a.seed)

    with open(GOLDEN) as f:
        goldens = {s["name"]: s for s in json.load(f)["samples"]}

    names = [n for n in a.samples.split(",") if n and n in goldens]
    if not names:
        print("no usable samples; run generate_goldens.py first")
        return 2

    # Encode once so request cost is the server's, not ours.
    urls = {n: data_url(os.path.join(DEMO, goldens[n]["image"])) for n in names}
    buckets = {n: goldens[n]["bucket"] for n in names}

    first_answer: dict[str, str] = {}
    failures: list[str] = []
    latencies: list[float] = []

    print(f"[order] seed={a.seed}, {a.passes} pass(es) over {len(names)} bucket(s)")
    for p in range(a.passes):
        order = names[:]
        rng.shuffle(order)
        print(f"  pass {p}: {' -> '.join(f'{n}({buckets[n]})' for n in order)}")
        for name in order:
            try:
                text, dt = ocr_request(base, urls[name], a.max_tokens, a.timeout)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"    {name:18s} FAILED ({type(e).__name__}: {e})")
                failures.append(f"{name}@pass{p}")
                continue
            latencies.append(dt)
            key = normalize(text)
            if name not in first_answer:
                first_answer[name] = key
                print(f"    {name:18s} bucket={buckets[name]:5d} {dt:6.1f}s  (reference)")
            elif key == first_answer[name]:
                print(f"    {name:18s} bucket={buckets[name]:5d} {dt:6.1f}s  match")
            else:
                print(f"    {name:18s} bucket={buckets[name]:5d} {dt:6.1f}s  DRIFT")
                failures.append(f"{name}@pass{p}:drift")

    # ---- steady-state depth on the cheapest image ------------------------
    if a.sequential:
        probe = min(names, key=lambda n: buckets[n])
        print(f"[depth] {a.sequential} sequential requests on {probe} (bucket {buckets[probe]})")
        drift = 0
        seq_lat = []
        for i in range(a.sequential):
            try:
                text, dt = ocr_request(base, urls[probe], a.max_tokens, a.timeout)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"    request {i} FAILED ({type(e).__name__}: {e})")
                failures.append(f"sequential@{i}")
                break
            seq_lat.append(dt)
            if normalize(text) != first_answer.get(probe, normalize(text)):
                drift += 1
        if seq_lat:
            print(
                f"    completed {len(seq_lat)}/{a.sequential}  "
                f"min={min(seq_lat):.2f}s median={statistics.median(seq_lat):.2f}s max={max(seq_lat):.2f}s"
            )
        if drift:
            print(f"    DRIFT on {drift} of {len(seq_lat)} responses")
            failures.append("sequential:drift")
        latencies.extend(seq_lat)

    print("\n============== SERVE HARDENING ==============")
    print(f"endpoint  : {base}")
    print(f"requests  : {len(latencies)} completed")
    if latencies:
        print(
            f"latency   : min={min(latencies):.2f}s median={statistics.median(latencies):.2f}s "
            f"max={max(latencies):.2f}s"
        )
    print(f"result    : {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures[:8])}")
    print("=============================================")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
