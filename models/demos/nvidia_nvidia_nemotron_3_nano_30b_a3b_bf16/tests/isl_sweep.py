#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ISL sweep benchmark: send N serial requests at a given ISL and measure TPOT.

Usage:
    python isl_sweep.py --isl 512 --n 100 --osl 32
    python isl_sweep.py --isl 128 512 2048 8192 --n 10
"""

import argparse
import time

import requests

HOST = "localhost"
PORT = 8000
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
VOCAB_SIZE = 131072
TOKEN_ID = 42  # a single repeated token ID for ISL control


def make_prompt(isl: int) -> str:
    """Return a string of exactly isl tokens using the word 'the'."""
    # Each word "the " ≈ 1 token in most tokenizers.  We over-generate and
    # rely on max_prompt_length clamping, OR we use a token-count exact approach
    # by repeating a single character token.  Use a numeric sequence for safety.
    words = "the " * (isl + 10)
    return words[: isl * 4]  # rough upper bound, truncated server-side


def make_token_ids_prompt(isl: int) -> list[int]:
    """Return exactly isl token IDs (all the same benign token)."""
    return [TOKEN_ID] * isl


def send_request(isl: int, osl: int, timeout_s: float = 300.0):
    """Send a chat completion request; return (ok, tpot_ms, n_output_tokens)."""
    # Use token IDs approach via a placeholder prompt that forces exact length.
    # Actually send a text prompt and rely on truncation to isl tokens.
    prompt = " ".join(["the"] * isl)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    url = f"http://{HOST}:{PORT}/v1/chat/completions"

    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        elapsed = time.monotonic() - t0
        if resp.status_code != 200:
            return False, 0.0, 0
        data = resp.json()
        n_out = data["usage"]["completion_tokens"]
        if n_out <= 0:
            return False, 0.0, 0
        tpot_ms = elapsed / n_out * 1000.0
        return True, tpot_ms, n_out
    except Exception as e:
        print(f"    ERROR: {e}")
        return False, 0.0, 0


def run_sweep(isls, n_requests, osl):
    for isl in isls:
        print(f"\n{'='*60}")
        print(f"ISL={isl}, OSL={osl}, N={n_requests}")
        print(f"{'='*60}")
        successes = 0
        tpot_sum = 0.0
        for i in range(n_requests):
            ok, tpot_ms, n_out = send_request(isl, osl)
            status = "OK" if ok else "FAIL"
            print(f"  req {i+1:3d}/{n_requests}: {status}  tpot={tpot_ms:.1f}ms  n_out={n_out}")
            if ok:
                successes += 1
                tpot_sum += tpot_ms
        avg_tpot = tpot_sum / successes if successes else float("nan")
        avg_toks = 1000.0 / avg_tpot if avg_tpot > 0 else 0.0
        print(f"\n  RESULT ISL={isl}: {successes}/{n_requests} OK  avg_tpot={avg_tpot:.1f}ms ({avg_toks:.2f} tok/s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, nargs="+", default=[512])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--osl", type=int, default=32)
    args = ap.parse_args()
    run_sweep(args.isl, args.n, args.osl)
