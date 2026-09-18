# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exercise a running PaddleOCR-VL server over the OpenAI-compatible API.

Checks the things that distinguish "the process is up" from "the endpoint is
usable": health, that the model is advertised, that an image request comes back
with the right text, and that repeated requests at *different resolutions* keep
working. That last one is the important one. The vision tower compiles a program
set per padded patch count, so the failure this catches is a request succeeding
and the next one at a new bucket hanging or returning garbage, which is the
classic symptom of a compile landing while a trace is parked (tt-metal #48536).

Scored against the recorded HuggingFace goldens where a sample has one, so a
regression shows up as text drift rather than only as an exception.

Run (server on :8100)::

    python models/demos/blackhole/paddleocr_vl/tests/serve_smoke.py --port 8100
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request

from PIL import Image

DEMO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
GOLDEN = os.path.join(DEMO, "golden", "hf_goldens.json")
MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PROMPT = "OCR:"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def _get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode()


def _post(url: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ocr_request(base: str, image_path: str, max_tokens: int, prompt: str = OCR_PROMPT):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    t0 = time.time()
    status, body = _post(f"{base}/v1/chat/completions", payload)
    dt = time.time() - t0
    choice = body["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason"), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--samples",
        default="sign_exit,receipt_cafe,table_quarterly,page_manual",
        help="comma-separated golden names; defaults to one per vision bucket",
    )
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"

    failures = []

    # ---- 1. health -------------------------------------------------------
    try:
        status, _ = _get(f"{base}/health")
        print(f"[1] GET /health            -> {status}")
        if status != 200:
            failures.append("health")
    except urllib.error.URLError as e:
        print(f"[1] GET /health            -> UNREACHABLE ({e})")
        return 2

    # ---- 2. model advertised --------------------------------------------
    status, body = _get(f"{base}/v1/models")
    ids = [m["id"] for m in json.loads(body).get("data", [])]
    ok = MODEL in ids
    print(f"[2] GET /v1/models         -> {status} {ids} {'OK' if ok else 'MISSING'}")
    if not ok:
        failures.append("models")

    # ---- 3/4. one image per bucket, scored ------------------------------
    goldens = {}
    if os.path.exists(GOLDEN):
        with open(GOLDEN) as f:
            goldens = {s["name"]: s for s in json.load(f)["samples"]}

    names = [n for n in a.samples.split(",") if n]
    print(f"[3] OCR requests across {len(names)} resolution(s):")
    rows = []
    for name in names:
        g = goldens.get(name)
        if g is None:
            print(f"    {name:20s} SKIP (no golden recorded yet)")
            continue
        path = os.path.join(DEMO, g["image"])
        try:
            text, finish, dt = ocr_request(base, path, a.max_tokens)
        except Exception as e:  # noqa: BLE001 - surface any transport/server error verbatim
            print(f"    {name:20s} FAILED: {type(e).__name__}: {e}")
            failures.append(name)
            continue
        c = cer(g["hf_output"], text)
        rows.append((name, g["bucket"], dt, finish, c))
        flag = "OK" if c <= 0.02 else "DRIFT"
        print(f"    {name:20s} bucket={g['bucket']:5d} {dt:6.1f}s finish={finish:8s} CERvsHF={c*100:6.2f}% {flag}")
        if c > 0.02:
            failures.append(name)

    # ---- 5. repeat the first sample: same answer twice -------------------
    if rows:
        first = rows[0][0]
        path = os.path.join(DEMO, goldens[first]["image"])
        t1, _, _ = ocr_request(base, path, a.max_tokens)
        t2, _, _ = ocr_request(base, path, a.max_tokens)
        same = normalize(t1) == normalize(t2)
        print(f"[4] repeat determinism     -> {'OK' if same else 'DIVERGED'}")
        if not same:
            failures.append("determinism")

    print("\n================ SERVE SMOKE ================")
    print(f"endpoint : {base}")
    print(f"result   : {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    print("=============================================")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
