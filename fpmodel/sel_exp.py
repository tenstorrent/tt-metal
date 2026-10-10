"""Selection rules on out-of-fold predictions: how the pick is made from predicted times, not the times themselves.
Tests shrinking the learned correction and tie-breaks among near-tied candidates (winner's curse control).
usage: sel_exp.py OOF_PREFIX   (OOF_PREFIX_{base,final}.npy from resid_cv.py with the same SETS)"""
import sys

sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
from data import load, SETS, CAND, gm

pre = sys.argv[1]
for k in [k for k in SETS if k.endswith("_fin1") and "fin1" not in pre]:  # OOF files predating round 1
    SETS.pop(k)
d = M.annotate(load(list(SETS)))
d["pb"], d["pf"] = np.load(f"{pre}_base.npy"), np.load(f"{pre}_final.npy")
d["c"] = np.log(d.pf / d.pb)
leg = d[d.origin == "legacy"].groupby("problem_id").device_ns.min()
x = d[d.origin.isin(CAND) & ~d.mcprobe & d.pf.notna()].reset_index(drop=True)
x["legacy"] = x.problem_id.map(leg)
x["grp"] = np.where(
    x.set.str.contains("draws"), "draws", np.where(x.set.str.contains("designed"), "designed", "rounds")
)
x["kb"] = x.in0_block_w.fillna(0)


def pick(score, tie=None, delta=0.0):
    """argmin of score; with tie: among candidates within delta of the min score, argmin of tie"""
    s = pd.Series(np.log(score), index=x.index)
    m = s.groupby(x.problem_id).transform("min")
    if tie is None:
        return s.groupby(x.problem_id).idxmin()
    t = pd.Series(np.where(s <= m + np.log1p(delta), tie, np.inf), index=x.index)
    return t.groupby(x.problem_id).idxmin()


def report(label, idx):
    r = x.loc[idx.values]
    out = []
    for a in ("wh", "bh"):
        for gname in ("draws", "designed", "rounds"):
            q = r[(r.arch_ == a) & (r.grp == gname)]
            v = (q.device_ns / q.legacy).dropna()
            out.append(f"{a}-{gname} {gm(q.device_ns/q.best):.3f}/{int((v>1.05).sum()):3d}/{int((v>1.25).sum()):2d}")
    print(f"{label:34s} " + " | ".join(out), flush=True)


print("columns: arch-group regret-gm / >5% slower than legacy / >25%")
report("base argmin", pick(x.pb))
report("final argmin (current)", pick(x.pf))
for s in (0.5, 0.75):
    report(f"shrink {s}", pick(x.pb * np.exp(s * x.c)))
for dl in (0.01, 0.02, 0.03, 0.05):
    report(f"tie {dl:.0%}: shallowest kb", pick(x.pf, x.kb + 1e-6 * x.pf, dl))
    report(f"tie {dl:.0%}: base argmin", pick(x.pf, x.pb, dl))
    report(f"tie {dl:.0%}: smallest |corr|", pick(x.pf, np.abs(x.c) + 1e-9 * x.pf, dl))

# uncertainty-aware pick: penalise each candidate by the typical within-problem error of its (family, K-depth) group,
# estimated out of fold by problem halves so a problem's own errors never set its penalty
x["kbb"] = pd.cut(x.kb, [-1, 1, 2, 4, 8, 16, 32, 1e9]).astype(str)
e = np.log(x.device_ns / x.pf)
x["w"] = (e - e.groupby(x.problem_id).transform("median")).abs()
half = x.problem_id.map(lambda s: hash(s) % 2)
sig = pd.Series(np.nan, index=x.index)
for h in (0, 1):
    tab = x[half != h].groupby(["arch_", "family", "kbb"]).w.quantile(0.75)
    m = half == h
    sig[m] = [tab.get(k, np.nan) for k in zip(x.arch_[m], x.family[m], x.kbb[m])]
sig = sig.fillna(sig.median())
for lam in (0.25, 0.5, 1.0):
    report(f"uncertainty penalty {lam}", pick(x.pf * np.exp(lam * sig)))
