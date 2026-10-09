"""Regret on fully swept device sets (every candidate timed): frozen model picks vs the best candidate, the rules and
legacy, plus within-problem ranking accuracy. usage: sweep_eval.py CONST_JSON SWEEP_CSV [SWEEP_CSV ...]"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
from data import CAND, gm

p = json.load(open(sys.argv[1]))
for name, (_, _, term) in M.CONSTANTS.items():
    if term and name not in p:
        M.OFF.add(term)
rows = []
for path in sys.argv[2:]:
    s = pd.read_csv(path, low_memory=False)
    s = s[s.status == "ok"].drop_duplicates(["case", "origin", "config"], keep="last").reset_index(drop=True)
    s["arch_"] = "wh"
    c = M.annotate(s[s.origin.isin(CAND)].reset_index(drop=True))
    c["pred"] = M.predict(M.geometry(c), p)
    leg = s[s.origin == "legacy"].groupby("case").device_ns.median()
    for case, g in c.groupby("case"):
        if case not in leg.index:
            continue
        best = g.device_ns.min()
        pk = g.loc[g.pred.idxmin()]
        rul = g[g.origin == "heuristic"].device_ns
        a, q = g.device_ns.to_numpy(), g.pred.to_numpy()
        i, j = np.triu_indices(len(a), 1)
        sel = np.abs(np.log(a[i] / a[j])) > np.log(1.05)
        rows.append(
            dict(
                set=os.path.basename(path),
                case=case,
                n=len(g),
                model=pk.device_ns / best,
                rules=rul.min() / best if len(rul) else np.nan,
                legacy=leg[case] / best,
                acc=((a[i] < a[j]) == (q[i] < q[j]))[sel].mean() if sel.any() else np.nan,
                fam=pk.family,
                best_fam=g.loc[g.device_ns.idxmin()].family,
            )
        )
R = pd.DataFrame(rows)
for s_, g in R.groupby("set"):
    v = g.model / g.legacy
    print(
        f"{s_}: {len(g)} problems | regret model {gm(g.model):.3f} (p90 {g.model.quantile(.9):.2f}, worst {g.model.max():.2f}) "
        f"rules {gm(g.rules.dropna()):.3f} legacy {gm(g.legacy):.3f} | model/legacy {gm(v):.3f}, >1.05 {int((v>1.05).sum())} | pair acc {g.acc.mean():.3f}"
    )
R.to_csv("/tmp/sweep_eval_last.csv", index=False)
