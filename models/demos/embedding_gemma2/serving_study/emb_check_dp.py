"""Correctness of every DP replica against the HF fp32 reference (emb_hf_ref.py), through the server.

Needs the server started with --debug-chip. Two passes:
  1. per chip, batch 1: each check text is sent as DP concurrent single-text requests on new connections (so they
     spread over front-ends, and least-loaded dispatch spreads them over each front-end's chips), repeated until every
     chip has embedded it. Gate: cosine(TT, HF) >= --gate on every (text, chip), and every chip seen for every text.
  2. list requests: 16 concurrent requests, each all texts x --copies/8, so every row of a list request is checked
     and the chips run at the server's batch setting. Same gate.
Usage: emb_check_dp.py URL REF_DIR OUT.json [--dp 32 --gate 0.99 --copies 64]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("ref_dir")
ap.add_argument("out")
ap.add_argument("--dp", type=int, default=32)
ap.add_argument("--gate", type=float, default=0.99)
ap.add_argument("--copies", type=int, default=64)
ap.add_argument("--rounds", type=int, default=30, help="pass 1: most rounds of DP concurrent requests per text")
a = ap.parse_args()

ref_dir = Path(a.ref_dir)
meta = json.load(open(ref_dir / "check_texts.json"))
texts, ntok = meta["texts"], meta["n_tokens"]
ref = np.load(ref_dir / "hf_ref.npy")
url = a.url.rstrip("/") + "/v1/embeddings"


def post(inp):
    r = requests.post(url, json={"input": inp}, timeout=600)
    r.raise_for_status()
    return r.json()["data"]


def cos(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


# pass 1: every text on every chip, batch 1. With several front-ends on one port (SO_REUSEPORT) a request reaches only
# its front-end's chips, so send concurrent single-text requests, each on a new connection, until all chips answered.
from concurrent.futures import ThreadPoolExecutor

per_chip = np.full((len(texts), a.dp), np.nan)
spread = []
with ThreadPoolExecutor(a.dp) as ex:
    for i, t in enumerate(texts):
        for _ in range(a.rounds):
            for data in ex.map(lambda _: post(t), range(a.dp)):
                d = data[0]
                c = cos(d["embedding"], ref[i])
                per_chip[i, d["chip"]] = c if np.isnan(per_chip[i, d["chip"]]) else min(c, per_chip[i, d["chip"]])
            if not np.isnan(per_chip[i]).any():
                break
        spread.append(int((~np.isnan(per_chip[i])).sum()))
missing = int(np.isnan(per_chip).sum())
p1_min = float(np.nanmin(per_chip))
worst_text = int(np.nanargmin(np.nanmin(per_chip, axis=1)))
chip_spread = float(np.nanmax(np.nanmax(per_chip, axis=1) - np.nanmin(per_chip, axis=1)))

# pass 2: list requests, several at once (spread over front-ends), each holding every text --copies // 8 times
inp = [texts[k % len(texts)] for k in range(len(texts) * max(1, a.copies // 8))]
with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(lambda _: post(inp), range(16)))
c2 = np.array([cos(d["embedding"], ref[k % len(texts)]) for data in results for k, d in enumerate(data)])
chips2 = sorted({d["chip"] for data in results for d in data})
data = [d for r in results for d in r]

res = dict(
    n_texts=len(texts),
    tokens_min_max=[min(ntok), max(ntok)],
    dp=a.dp,
    pass1_cells=int(per_chip.size),
    pass1_missing_cells=missing,
    pass1_every_text_hit_all_chips=all(s == a.dp for s in spread),
    pass1_min_cos=round(p1_min, 6),
    pass1_mean_cos=round(float(np.nanmean(per_chip)), 6),
    pass1_worst_text_index=worst_text,
    pass1_worst_text_tokens=ntok[worst_text],
    pass1_per_text_min_cos=[round(float(v), 6) for v in np.nanmin(per_chip, axis=1)],
    pass1_max_cross_chip_cos_spread=chip_spread,
    pass2_rows=len(data),
    pass2_chips_used=len(chips2),
    pass2_min_cos=round(float(c2.min()), 6),
    pass2_mean_cos=round(float(c2.mean()), 6),
    gate=a.gate,
)
res["pass"] = bool(missing == 0 and p1_min >= a.gate and c2.min() >= a.gate and res["pass1_every_text_hit_all_chips"])
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "pass1_per_text_min_cos"}, indent=1))
