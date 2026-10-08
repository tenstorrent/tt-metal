"""Fit the current best variant (model_x, MX/FIX as in the env) on all training data for one arch, then pick
(plain argmin) from an enumerated-candidates CSV. usage: MX=... FIX=... pick_x.py ENUM_CSV ARCH OUT_CSV"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model_x as M, fit, nocload
from data import load, SETS, KEY, CAND

fit.predict = M.predict
fit.PARAMS = M.PARAMS
fit.NAMES = list(M.PARAMS)
fit.geometry = M.geometry
fit.HI.update(eta_max=0.95, link_eff=1.0)
fit.LO.update(eta_max=0.3, link_eff=0.05)
for kv in filter(None, os.environ.get("FIX", "").split(",")):
    k, v = kv.split("=")
    fit.LO[k] = float(v) * 0.999
    fit.HI[k] = float(v) * 1.001
    M.PARAMS[k] = float(v)
src, arch, out = sys.argv[1:4]
d = load(list(SETS))
d = d[d.arch_ == arch].reset_index(drop=True)
if "noc" in M.MX:
    d["link_bytes"] = nocload.link_bytes(M.geometry(d), d)
p = (
    json.load(open(os.environ["CONST"])) if os.environ.get("CONST") else fit.fit(d, np.arange(len(d)))
)  # CONST: frozen constants file
json.dump(p, open(out.replace(".csv", "_constants.json"), "w"), indent=1)
e = pd.read_csv(src, low_memory=False).drop_duplicates(KEY, keep="last")
e = e[e.origin.isin(CAND) & (e.status == "ok")].reset_index(drop=True)
e["arch_"] = arch
if "noc" in M.MX:
    e["link_bytes"] = nocload.link_bytes(M.geometry(e), e)
e["pred"] = M.predict(M.geometry(e), p)
r = e.loc[e.groupby("case").pred.idxmin(), ["case", "config", "family", "origin", "pred"]]
r["pred_us"] = (r.pred / 1e3).round(2)
r.drop(columns="pred").to_csv(out, index=False)
print(len(r), "picks; same as rules:", int((r.origin == "heuristic").sum()), r.family.value_counts().to_dict())
