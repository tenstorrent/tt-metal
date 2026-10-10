"""K-depth ranking within a fixed tiling on a swept set: for each (problem, tiling) with several K depths timed, how
often the model's best K depth is the measured best, the regret of its K choice, and the per-step slope bias.
usage: kdepth_eval.py CONST_JSON SWEEP_CSV [ARCH]"""
import os, sys, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
from data import CAND

p = json.load(open(sys.argv[1]))
for name, (_, _, term) in M.CONSTANTS.items():
    if term and name not in p:
        M.OFF.add(term)
s = pd.read_csv(sys.argv[2], low_memory=False)
s = s[(s.status == "ok") & s.origin.isin(CAND)].drop_duplicates(["case", "config"], keep="last").reset_index(drop=True)
s["arch_"] = sys.argv[3] if len(sys.argv) > 3 else "wh"
s = M.annotate(s)
s["pred"] = M.predict(M.geometry(s), p)
key = [
    "case",
    "family",
    "per_core_M",
    "per_core_N",
    "out_block_h",
    "out_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "fuse_batch",
]
rows = []
for k, g in s.groupby(key, dropna=False):
    g = g.drop_duplicates("in0_block_w")
    if len(g) < 2:
        continue
    best = g.device_ns.min()
    pick = g.loc[g.pred.idxmin()]
    deeper = pick.in0_block_w > g.loc[g.device_ns.idxmin()].in0_block_w
    rows.append(
        dict(case=k[0], n=len(g), regret=pick.device_ns / best, right=pick.device_ns <= best * 1.02, deeper=deeper)
    )
r = pd.DataFrame(rows)
gm = lambda x: float(np.exp(np.log(x).mean()))
w = r[~r.right]
print(
    f"tilings with >=2 K depths: {len(r)} | K choice within 2% of best: {r.right.mean():.1%} | regret gm {gm(r.regret):.3f} p90 {r.regret.quantile(.9):.3f}"
    f" | wrong picks too deep: {int(w.deeper.sum())}, too shallow: {int((~w.deeper).sum())}"
)
