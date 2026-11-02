"""python3 -I fusedec_routes_stats.py <layers comma list> [n_rows_users=8] [steps=4] [seed=7] -- distinct experts per chip of the injected routes (replays the sampling of test_fusedec_round_prof._inject_routes)
and the distribution over 2000 random draws of (users x consecutive steps) of the same routes: mean / busiest-chip expert count per layer (sequential expert -> chip mapping of the production YAML)."""
import sys

import numpy as np
import torch

R0 = "/mnt/tt-data/ssinghal/dsv4-logs/moe_place/runs"


def load(paths):
    arrs = []
    for p in paths:
        r = torch.load(p)
        a = np.stack([np.stack([t.numpy() for t in st]) for st in r]).transpose(1, 0, 2, 3)  # [L, S, B, 6]
        arrs.append(a)
    S = min(a.shape[1] for a in arrs)
    return np.concatenate([a[:, :S] for a in arrs], axis=2)


chip = lambda e: ((e % 48) // 12) * 8 + e // 48


def counts(ids):
    cnt = {}
    for e in set(ids.flatten().tolist()):
        cnt.setdefault(chip(e), set()).add(e)
    n = [len(cnt.get(c, ())) for c in range(32)]
    return float(np.mean(n)), max(n)


layers = [int(x) for x in sys.argv[1].split(",")]
users = int(sys.argv[2]) if len(sys.argv) > 2 else 8
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
R = load([f"{R0}/cg1/routes.pt", f"{R0}/cq/routes.pt"])
S, Bu = R.shape[1], R.shape[2]
rng = np.random.default_rng(seed)
print("injected instance (same sampling order as the test):")
dump = {}
for lid in layers:
    us = rng.choice(Bu, size=min(users, Bu), replace=False)
    s0 = int(rng.integers(0, S - steps + 1))
    ids = R[lid][s0 : s0 + steps][:, us].transpose(1, 0, 2).reshape(-1, 6)
    dump[lid] = ids.tolist()
    m, mx = counts(ids)
    print(
        f"  layer {lid}: mean experts/chip {m:.2f}, busiest chip {mx}, distinct experts {len(set(ids.flatten().tolist()))}"
    )
if len(sys.argv) > 5:
    import json

    json.dump(dump, open(sys.argv[5], "w"))
print("distribution over 2000 draws (mean of mean / mean of max / 10th-90th pct of max):")
g = np.random.default_rng(1)
for lid in layers:
    ms, mxs = [], []
    for _ in range(2000):
        us = g.choice(Bu, size=users, replace=False)
        s0 = int(g.integers(0, S - steps + 1))
        m, mx = counts(R[lid][s0 : s0 + steps][:, us].reshape(-1, 6))
        ms.append(m)
        mxs.append(mx)
    print(
        f"  layer {lid}: mean {np.mean(ms):.2f}, busiest {np.mean(mxs):.2f} (p10 {np.percentile(mxs, 10):.0f}, p90 {np.percentile(mxs, 90):.0f})"
    )
