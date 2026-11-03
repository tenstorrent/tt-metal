"""Closed-loop load on /v1/embeddings: C clients, each sends one single-text request after the previous reply, for D s.

Usage: emb_load.py URL out.json TOKENS CONC [CONC ...]   env EMB_LOAD_SECONDS (default 30), EMB_LOAD_WARMUP_S (5).
Prints one JSON row per concurrency: req/s, p50/p90/p99 ms, errors.
"""

import json
import os
import sys
import threading
import time

import requests

from emb_common import MODEL_ID, load_tokenizer, make_texts

url, out_path, n_tok = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3])
concs = [int(c) for c in sys.argv[4:]]
seconds = float(os.environ.get("EMB_LOAD_SECONDS", "30"))
warmup = float(os.environ.get("EMB_LOAD_WARMUP_S", "5"))
headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {os.environ.get('API_KEY', 'your-secret-key')}"}
texts = make_texts(load_tokenizer(), n_tok, 64)


def client(i, t_start, t_end, lat, errs):
    sess = requests.Session()
    k = i
    while time.perf_counter() < t_end:
        body = {"model": MODEL_ID, "input": texts[k % len(texts)]}
        k += 7
        a = time.perf_counter()
        try:
            sess.post(f"{url}/v1/embeddings", headers=headers, json=body, timeout=120).raise_for_status()
        except Exception:
            errs.append(1)
            continue
        b = time.perf_counter()
        if a >= t_start and b <= t_end:
            lat.append((b - a) * 1e3)


rows = []
for c in concs:
    lat, errs = [], []
    t0 = time.perf_counter()
    t_start, t_end = t0 + warmup, t0 + warmup + seconds
    ths = [threading.Thread(target=client, args=(i, t_start, t_end, lat, errs)) for i in range(c)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else None
    row = dict(tokens=n_tok, conc=c, seconds=seconds, n=len(lat), req_s=len(lat) / seconds,
               p50_ms=q(0.5), p90_ms=q(0.9), p99_ms=q(0.99), errors=len(errs))
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(dict(url=url, rows=rows), open(out_path, "w"), indent=1)
