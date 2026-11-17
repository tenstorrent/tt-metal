#!/usr/bin/env python3
"""Concurrent counting-continuation check: batched prefills of mixed lengths, then prefix-cached extensions."""
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 32


def complete(text, n=6):
    body = {"model": "llama70b", "prompt": text, "max_tokens": n, "temperature": 0, "top_k": 32}
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    return r["choices"][0]["text"], r["usage"]["prompt_tokens"], (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")


def counting(k):
    return ", ".join(str(i) for i in range(1, k + 1)) + ","


def one(job):
    header, k = job
    out, pt, cached = complete(header + counting(k))
    return k, pt, cached, out.strip().startswith(str(k + 1)), out.strip()[:12]


total = bad = 0
for rnd in range(ROUNDS):
    ks = [random.choice([10, 20, 25, 30, 40, 60, 90, 150, 300, 700, 1500]) + random.randint(0, 9) for _ in range(CONC)]
    headers = [f"Session {random.randint(10**8, 10**9)}. Continue the list.\n" for _ in range(CONC)]
    with ThreadPoolExecutor(CONC) as ex:
        first = list(ex.map(one, zip(headers, ks)))
        ext = list(ex.map(one, zip(headers, [k + random.randint(2, 40) for k in ks])))
    for tag, res in (("fresh", first), ("cached", ext)):
        wrong = [r for r in res if not r[3]]
        total += len(res)
        bad += len(wrong)
        hits = sum(1 for r in res if r[2])
        print(f"round {rnd} {tag:6s} wrong {len(wrong)}/{len(res)} cache_hits={hits} "
              + " ".join(f"[k={r[0]} pt={r[1]} c={r[2]} {r[4]!r}]" for r in wrong[:6]), flush=True)
print(f"TOTAL wrong {bad}/{total}", flush=True)
