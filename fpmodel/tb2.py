"""Reload-aware tie-break, judged on the training sets only (out-of-fold predictions).
Among candidates predicted within MARGIN of the best: when partials are spilled and reloaded every K step
(packer_l1_acc off), a shallow K block multiplies the reloads, so the shallow-K preference is not safe there."""
import sys, os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *

MARGIN = 0.05


def choose(g, spill_rule):
    """g: one problem's candidates with pred. Returns the index of the pick."""
    best = g.pred.min()
    near = g[g.pred <= best * (1 + MARGIN)]
    spill = g.packer_l1_acc.fillna(0).iloc[0] == 0
    kb = near.in0_block_w.fillna(1)
    if spill and spill_rule == "argmin":
        return g.pred.idxmin()
    if spill and spill_rule == "deep":
        return near.assign(k=-kb).sort_values(["k", "pred"]).index[0]
    if spill and spill_rule == "fp32argmin" and g.fp32_acc.fillna(0).iloc[0] == 1:
        return g.pred.idxmin()
    return near.assign(k=kb).sort_values(["k", "pred"]).index[0]


def score(d, rule):
    s = np.full(len(d), 1e12)
    c = d[d.origin.isin(CAND)]
    for _, g in c.groupby("problem_id", sort=False):
        s[choose(g, rule)] = 0.0
    return s


if __name__ == "__main__":
    d = load(list(SETS))
    d["pred"] = np.load(sys.argv[1] if len(sys.argv) > 1 else "pred_cv.npy")
    spill = d.groupby("problem_id").packer_l1_acc.first().fillna(0) == 0
    print(
        "problems with spills every K step:",
        d[d.problem_id.isin(spill[spill].index)].groupby("set").problem_id.nunique().to_dict(),
    )
    for rule in ("shallow", "argmin", "fp32argmin", "deep"):
        r = evaluate(d, score(d, rule), f"spill rule: {rule}")
        sp = r[r.problem_id.isin(spill[spill].index)]
        print(
            f"   spill problems only: n={len(sp)} regret {gm(sp.regret):.3f}  >5% slower than legacy {(sp.vs_legacy > 1.05).sum()}"
            f"  >25% {(sp.vs_legacy > 1.25).sum()}"
        )
