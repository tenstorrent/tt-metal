"""HTTP client for /v1/embeddings (FastAPI template or tt-media-server): concurrency 1, timed end to end.

Usage: emb_client.py URL out.json [reps] [bare.json]  -- bare.json (from emb_bare.py) adds a cosine check per point.
EMB_FANOUT=1: a batch of B texts is sent as B single-text requests in flight at once (wall time, first send to last
reply); tt-media-server accepts one string per request and batches concurrent requests itself.
EMB_BATCHED_ONLY=1: only the batch > 1 points.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from emb_common import MODEL_ID, POINTS, load_tokenizer, make_texts

url, out_path = sys.argv[1].rstrip("/"), sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
golden = json.load(open(sys.argv[4]))["golden"] if len(sys.argv) > 4 else {}
headers = {"Content-Type": "application/json"}
headers["Authorization"] = f"Bearer {os.environ.get('API_KEY', 'your-secret-key')}"  # media server default key
tok = load_tokenizer()
sess = requests.Session()
fanout = os.environ.get("EMB_FANOUT") == "1"
points = [p for p in POINTS if p[0] > 1] if os.environ.get("EMB_BATCHED_ONLY") == "1" else POINTS
pool = ThreadPoolExecutor(32) if fanout else None
sessions = [requests.Session() for _ in range(32)] if fanout else None


def _post_one(i, text):
    resp = sessions[i].post(f"{url}/v1/embeddings", headers=headers, json={"model": MODEL_ID, "input": text}, timeout=600)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


rows = []
for batch, n_tok in points:
    texts = make_texts(tok, n_tok, batch)
    body = {"model": MODEL_ID, "input": texts if batch > 1 else texts[0]}
    lat = []
    for r in range(reps + 3):
        a = time.perf_counter()
        if fanout and batch > 1:
            embs = list(pool.map(_post_one, range(batch), texts))
        else:
            resp = sess.post(f"{url}/v1/embeddings", headers=headers, json=body, timeout=600)
            resp.raise_for_status()
        b = time.perf_counter()
        if r >= 3:
            lat.append((b - a) * 1e3)
    if not (fanout and batch > 1):
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        embs = [d["embedding"] for d in data]
    cos = None
    g = golden.get(f"{batch}x{n_tok}")
    if g:
        import math

        dot = lambda u, v: sum(x * y for x, y in zip(u, v))
        cos = min(dot(u, v) / math.sqrt(dot(u, u) * dot(v, v)) for u, v in zip(embs, g))
    lat.sort()
    row = dict(batch=batch, tokens=n_tok, reps=reps, fanout=fanout and batch > 1, n_out=len(embs), dim=len(embs[0]),
               e2e_ms_p50=lat[len(lat) // 2], e2e_ms_p90=lat[int(len(lat) * 0.9)], e2e_ms_min=lat[0], min_cos_vs_bare=cos)
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(dict(url=url, rows=rows), open(out_path, "w"), indent=1)
