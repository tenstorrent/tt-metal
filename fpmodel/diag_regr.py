"""For given cases: base and residual-corrected predictions of the model's pick, the legacy config and the rules' pick,
next to the measured times (same session). Shows whether a regression is a ranking error of the base or the residual.
usage: EXTRA=wfam diag_regr.py ENUM_CSV TIMED_CSV ARCH CONST_JSON RESID_LGB CASE_SUBSTR [...]"""
import sys, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R
from data import KEY

enum, timed, arch, cj, lg = sys.argv[1:6]
pats = sys.argv[6:]
p = json.load(open(cj))
for name, (_, _, term) in M.CONSTANTS.items():
    if term and name not in p:
        M.OFF.add(term)
e = pd.read_csv(enum, low_memory=False).drop_duplicates(KEY, keep="last")
e = e[(e.status == "ok") & e.per_core_M.notna() & e.family.isin(["2d", "1d_in0", "1d_in1", "reuse"])]
e = e[e.case.map(lambda c: any(s in c for s in pats))].reset_index(drop=True)
e["arch_"] = arch
e = M.annotate(e)
b, g, pr = R.base_parts(e, p)
e["base"] = b
e["pred"] = b * np.exp(lgb.Booster(model_file=lg).predict(R.features(e, g, pr)))
norm = lambda s: str(s).replace(" ", "")
e["cfg"] = e.config.map(norm)
t = pd.read_csv(timed, low_memory=False)
t = t[t.status == "ok"].assign(cfg=lambda z: z.config.map(norm))
pd.set_option("display.width", 250)
for c, q in e.groupby("case"):
    tt = t[t.case == c]
    rows = []
    for o in ["legacy", "heuristic", "model", "v18r9"]:
        r = tt[tt.origin == o]
        if not len(r):
            continue
        r = r.iloc[-1]
        m = q[q.cfg == r.cfg]
        if not len(m):
            m = q[q.origin == o]
        m = m.iloc[0] if len(m) else None
        rows.append(
            dict(
                who=o,
                fam=r.family,
                kb=r.in0_block_w,
                sb=f"{r.out_subblock_h}x{r.out_subblock_w}",
                blk=f"{r.out_block_h}x{r.out_block_w}",
                grid=f"{r.grid_x}x{r.grid_y}",
                meas_us=r.device_ns / 1e3,
                base_us=None if m is None else m.base / 1e3,
                pred_us=None if m is None else m.pred / 1e3,
            )
        )
    d = pd.DataFrame(rows)
    rank = q.pred.rank(method="first")
    print(f"== {c[:70]}  M={q.M.iloc[0]} K={q.K.iloc[0]} N={q.N.iloc[0]} b={q.batch.iloc[0]}  configs={len(q)}")
    print(d.round(1).to_string(index=False))
