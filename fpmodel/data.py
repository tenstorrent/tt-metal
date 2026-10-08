"""Sweep data and the selection metric for the first-principles matmul model.

A problem is one matmul (shapes, dtypes, memory, compute config). Each problem has many timed program configs.
The selector sees the candidates (enumerated configs + the rules' own pick) and picks one; regret = its device time
over the best candidate's. Legacy rows are never candidates or fitting data, only the acceptance yardstick.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd

WORK = "/localdev/rmiller/mm-oob-model-work"
SETS = {  # name -> (path, arch); read-only inputs
    "wh_designed": (f"{WORK}/frozen/wh_1505.csv", "wh"),
    "wh_target": (f"{WORK}/device/target_wh.csv", "wh"),
    "bh_designed": (f"{WORK}/frozen/bh_0200.csv", "bh"),
}
KEY = ["problem_id", "origin", "config"]
CAND = ("enumerated", "heuristic")
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def load(names):
    parts = []
    for n in names:
        path, arch = SETS[n]
        x = pd.read_csv(path, low_memory=False)
        x["set"], x["arch_"] = n, arch
        x["problem_id"] = n + ":" + x.problem_id.astype(str)
        parts.append(x.drop_duplicates(KEY, keep="last"))
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.status == "ok") & (d.device_ns > 0)].reset_index(drop=True)
    cand = d.origin.isin(CAND)
    d["best"] = d.problem_id.map(d[cand].groupby("problem_id").device_ns.min())
    has_rules = set(d.problem_id[d.origin == "heuristic"])  # compare on problems where the rules' pick was timed
    return d[d.best.notna() & d.problem_id.isin(has_rules)].reset_index(drop=True)


def evaluate(d, pred, label, min_best_us=0.0, quiet=False):
    """pred: predicted time aligned with d (lower = better). Picks the argmin candidate per problem."""
    x = d.assign(pred=pred)
    out = []
    for p, g in x.groupby("problem_id", sort=False):
        c = g[g.origin.isin(CAND)]
        if c.best.iloc[0] < min_best_us * 1e3:
            continue
        pick = c.loc[c.pred.idxmin()]
        rules = c[c.origin == "heuristic"].device_ns
        leg = g[g.origin == "legacy"].device_ns
        out.append(
            dict(
                problem_id=p,
                set=g.set.iloc[0],
                best=c.best.iloc[0],
                pick=pick.device_ns,
                pick_cfg=pick.config,
                rules=rules.min() if len(rules) else np.nan,
                legacy=leg.min() if len(leg) else np.nan,
            )
        )
    r = pd.DataFrame(out)
    r["regret"] = r.pick / r.best
    r["rules_regret"] = r.rules / r.best
    r["vs_legacy"] = r.pick / r.legacy
    r["rules_vs_legacy"] = r.rules / r.legacy
    if not quiet:
        for s, g in r.groupby("set", sort=False):
            print(
                f"{label:28s} {s:12s} n={len(g):4d} regret {gm(g.regret):.3f} (rules {gm(g.rules_regret.dropna()):.3f})"
                f"  worst {g.regret.max():.2f}  vs legacy {gm(g.vs_legacy.dropna()):.3f}"
                f"  >5% slower than legacy {int((g.vs_legacy > 1.05).sum()):3d} (rules {int((g.rules_vs_legacy > 1.05).sum())})"
                f"  worst/legacy {g.vs_legacy.max():.2f}"
            )
    return r


def rank_quality(d, pred):
    """Within-problem quality of the time prediction: pairwise order accuracy (pairs > 5% apart) and median |log err|."""
    acc, n = 0, 0
    x = d.assign(pred=pred)
    for _, g in x[x.origin.isin(CAND)].groupby("problem_id"):
        t, p = np.log(g.device_ns.to_numpy()), np.log(g.pred.to_numpy())
        dt, dp = t[:, None] - t[None, :], p[:, None] - p[None, :]
        m = dt > np.log(1.05)
        acc += int((dp[m] > 0).sum())
        n += int(m.sum())
    err = np.abs(np.log(x.pred / x.device_ns))
    return acc / max(n, 1), float(np.median(err)), float(np.quantile(err, 0.9))
