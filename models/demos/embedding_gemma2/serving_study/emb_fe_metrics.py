"""Collect /metrics from every front-end of a --reuse-port DP server (each answers for its own chips).

Polls on new connections until it has seen `--frontends` distinct chip sets. With --reset each front-end's counters
are cleared as it is seen (call before a load point); without it, prints the merged texts-per-forward histogram (the
batch the chips actually ran) and per-chip text counts (call after the point).
Usage: emb_fe_metrics.py URL [--frontends 8] [--reset] [--out FILE.json]
"""

import argparse
import json

import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--frontends", type=int, default=8)
ap.add_argument("--reset", action="store_true")
ap.add_argument("--out")
ap.add_argument("--tries", type=int, default=400)
a = ap.parse_args()

seen = {}
for _ in range(a.tries):
    m = requests.get(
        a.url.rstrip("/") + "/metrics",
        params={"reset": "true"} if a.reset else None,
        headers={"Connection": "close"},
        timeout=10,
    ).json()
    key = ",".join(map(str, m.get("chips", [])))
    if key and key not in seen:
        seen[key] = m
        if len(seen) == a.frontends:
            break
if a.reset:
    print(json.dumps({"reset_frontends": len(seen)}))
else:
    hist, per_chip = {}, {}
    for key, m in seen.items():
        for b, n in m.get("texts_per_forward", {}).items():
            hist[int(b)] = hist.get(int(b), 0) + n
        for c, n in zip(m["chips"], m["texts_per_chip"]):
            per_chip[str(c)] = n
    forwards = sum(hist.values())
    texts = sum(b * n for b, n in hist.items())
    res = dict(
        frontends=len(seen),
        forwards=forwards,
        texts=texts,
        mean_batch=round(texts / forwards, 2) if forwards else None,
        max_batch=max(hist) if hist else None,
        texts_in_b32=round(sum(b * n for b, n in hist.items() if b >= 32) / texts, 3) if texts else None,
        hist=dict(sorted(hist.items())),
        texts_per_chip_min_max=[min(per_chip.values()), max(per_chip.values())] if per_chip else None,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "hist"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
