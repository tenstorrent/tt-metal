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
    "bh_designed": (
        f"{WORK}/frozen/bh_designed_all2.csv",
        "bh",
    ),  # BH designed_all2: 2051 problems, full sweeps (superset of the 153 in bh_0200.csv)
    # full sweeps of fresh problems, added once their draws had served as clean tests (active learning)
    "wh_fresh1": (f"{WORK}/frozen/wh_fresh1_sweep.csv", "wh"),  # 59 swept draw-1 problems
    "wh_miss2": (f"{WORK}/frozen/wh_miss_draw2.csv", "wh"),  # draw-2 problems where v6 was > 3% slower than legacy
    "wh_sharded": (f"{WORK}/frozen/wh_sharded_sweep.csv", "wh"),  # sharded-input problems of draws 2 and 3
    # fresh draws whose clean tests are done (legacy, rules and the then-current model's pick per problem)
    "wh_kdepth": (
        f"{WORK}/frozen/wh_kdepth_full.csv",
        "wh",
    ),  # full sweeps of 66 shapes near the suite's K-depth misses
    # coverage round 1: the least-covered problems of a pool of suite-shape neighbours, contenders only (v18r2)
    "wh_cov1": (f"{WORK}/frozen/wh_cov1.csv", "wh"),
    # coverage round 2: sibling sweeps (K block depth, subblock) in the regimes of v18r2's remaining suite misses (v18r3)
    "wh_cov2": (f"{WORK}/frozen/wh_cov2.csv", "wh"),
    "wh_cov3": (f"{WORK}/frozen/wh_cov3.csv", "wh"),  # round 3: full sweeps near v18r3's confirmed suite misses (v18r4)
    "wh_cov4": (f"{WORK}/frozen/wh_cov4.csv", "wh"),  # round 4: full sweeps of random single-K-tile shapes (v18r6)
    "wh_wbench": (
        f"{WORK}/frozen/wh_wbench.csv",
        "wh",
    ),  # write-rate sweep: K=32 LoFi, output 1K-1M rows x 32-4096 cols, every config (v18r8)
    "bh_draws": (
        [
            f"{WORK}/frozen/bh_draws/{f}"
            for f in (
                "freshbh_timed.csv",
                "freshbh2_timed.csv",
                "freshbh3_timed.csv",
                "freshbh4_timed.csv",
                "freshbh5_timed.csv",
                "freshbh6_timed.csv",
                "freshbh7_timed.csv",
                "freshbh6_v15chg_timed.csv",
                "freshbh7_v17chg_timed.csv",
            )
        ],
        "bh",
    ),  # BH random draws 1-7, clean tests done
    "bh_kt1": (f"{WORK}/frozen/bh_kt1.csv", "bh"),  # BH: full sweeps of the 64 single-K-tile shapes
    "bh_wbench": (f"{WORK}/frozen/bh_wbench.csv", "bh"),  # BH: write-rate sweep, 75 shapes
    "bh_cov1": (f"{WORK}/frozen/bh_cov1.csv", "bh"),  # BH coverage round 1: 200 least-covered pool problems, contenders
    "wh_draws": ([f"{WORK}/frozen/wh_draw{k}_timed.csv" for k in (2, 3, 4, 5)], "wh"),
    # finalist round 1 (v18r10): new problems, v18r9's top 8 within 15% of its best, legacy and rules
    "wh_fin1": (f"{WORK}/frozen/wh_fin1.csv", "wh"),
    "bh_fin1": (f"{WORK}/frozen/bh_fin1.csv", "bh"),
    "wh_fin2": (f"{WORK}/frozen/wh_fin2.csv", "wh"),
    "bh_fin2": (f"{WORK}/frozen/bh_fin2.csv", "bh"),
}
# MultiCore probes: the MultiCore config timed on a set's own problems (origin "multicore"), joined unless MC_PROBES=0 (v18r4 on)
MC_PROBES = {"wh_designed": f"{WORK}/device/multicore_wh.csv", "bh_designed": f"{WORK}/bh/bh_designed_mc.csv"}
KEY = ["problem_id", "origin", "config"]
CAND = ("enumerated", "heuristic")
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def load(names):
    parts = []
    for n in names:
        path, arch = SETS[n]
        paths = path if isinstance(path, list) else [path]
        x = pd.concat(
            [
                pd.read_csv(q, low_memory=False).assign(
                    problem_id=lambda z, q=q: (q.split("/")[-1] + ":" if len(paths) > 1 else "")
                    + z.problem_id.astype(str)
                )
                for q in paths
            ],
            ignore_index=True,
        )
        x["mcprobe"] = False
        if os.environ.get("MC_PROBES", "1") == "1" and n in MC_PROBES:
            m = pd.read_csv(MC_PROBES[n], low_memory=False)
            m = m[m.problem_id.astype(str).isin(set(x.problem_id.astype(str)))]
            x = pd.concat([x, m.assign(origin="enumerated", family="multicore", mcprobe=True)], ignore_index=True)
        x["set"], x["arch_"] = n, arch
        if isinstance(path, list):  # timed draws: the model's pick is one of that problem's candidates
            x["origin"] = x.origin.replace({"model": "enumerated"})
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
