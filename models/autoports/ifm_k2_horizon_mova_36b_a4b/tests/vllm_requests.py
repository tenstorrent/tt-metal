# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small HTTP-only inner-loop requests; reduced layers are not quality evidence."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--timeout", type=int, default=120)
a = p.parse_args()
for attempt in range(60):
    try:
        if requests.get("http://localhost:8000/health", timeout=1).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Server not ready")


def run(n):
    started = time.time()
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "IFM/K2-Horizon-MoVA-36B-A4B",
            "prompt": [42] * n,
            "max_tokens": 4,
            "temperature": 0,
            "ignore_eos": True,
        },
        timeout=a.timeout,
    )
    return dict(length=n, status=response.status_code, seconds=time.time() - started, body=response.json())


results = [run(129)]
if results[0]["status"] == 200:
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        results += list(pool.map(run, [130, 131, 513]))
Path(a.output).write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
if any(r["status"] != 200 for r in results):
    raise SystemExit(1)
