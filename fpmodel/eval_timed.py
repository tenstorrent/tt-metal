"""Prediction error on every timed config of a timed set (legacy, rules, model picks), for a frozen model.
usage: EXTRA=wfam eval_timed.py ENUM_CSV TIMED_CSV ARCH CONST_JSON RESID_LGB [OFFSET_LGB]"""
import sys, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R
from data import KEY

enum, timed, arch, cj, lg = sys.argv[1:6]
p = json.load(open(cj))
for name, (_, _, term) in M.CONSTANTS.items():
    if term and name not in p:
        M.OFF.add(term)
e = pd.read_csv(enum, low_memory=False).drop_duplicates(KEY, keep="last")
e = e[(e.status == "ok") & e.per_core_M.notna() & e.family.isin(["2d", "1d_in0", "1d_in1", "reuse"])]
e = e.reset_index(drop=True).assign(arch_=arch)
e = M.annotate(e)
b, g, pr = R.base_parts(e, p)
f = R.features(e, g, pr)
e["pred"] = b * np.exp(lgb.Booster(model_file=lg).predict(f))
if len(sys.argv) > 6:
    o = lgb.Booster(model_file=sys.argv[6]).predict(R.problem_features(e, f))
    cand = e.origin.isin(["enumerated", "heuristic"]).to_numpy()
    off = pd.Series(np.where(cand, o, np.nan)).groupby(e.case.to_numpy()).transform("median").to_numpy()
    e["pred"] *= np.exp(off)
norm = lambda s: s.astype(str).str.replace(" ", "")
e["cfg"] = norm(e.config)
t = pd.read_csv(timed, low_memory=False)
t = t[t.status == "ok"].assign(cfg=lambda z: norm(z.config))
j = t.merge(e[["case", "cfg", "pred"]].drop_duplicates(["case", "cfg"]), on=["case", "cfg"])
j["err"] = np.log(j.pred / j.device_ns)
for o, q in list(j.groupby("origin")) + [("all", j)]:
    a = q.err.abs()
    print(
        f"{o:10s} n={len(q):4d} median {100*a.median():.1f}% p90 {100*a.quantile(.9):.1f}% within 5% {100*(a<np.log(1.05)).mean():.0f}%"
    )
