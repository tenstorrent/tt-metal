# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Targeted HTTP request probe; server lifecycle belongs to the shared runner."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", default="129")
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sampling", default='{"temperature":0}')
    args = parser.parse_args()
    sampling = json.loads(args.sampling)

    def request(length):
        payload = {
            "model": "ornith-ai/Ornith-1.5-9B",
            "prompt": [9707 + i % 13 for i in range(length)],
            "max_tokens": args.output_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
            **sampling,
        }
        started = time.monotonic()
        response = requests.post("http://localhost:8000/v1/completions", json=payload, timeout=300)
        return {
            "length": length,
            "status": response.status_code,
            "seconds": time.monotonic() - started,
            "request": payload,
            "response": response.json(),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(request, [int(x) for x in args.lengths.split(",")]))
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print([(r["length"], r["status"], round(r["seconds"], 3)) for r in results])
    if any(r["status"] != 200 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
