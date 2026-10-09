"""Pick with model7 (plain argmin) from an enumerated-candidates CSV, using frozen constants.
usage: pick7.py ENUM_CSV ARCH OUT_CSV [CONST_JSON (default fitted_v7_<arch>.json)]"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import pandas as pd
import model7 as M
from data import KEY, CAND

src, arch, out = sys.argv[1:4]
p = json.load(open(sys.argv[4] if len(sys.argv) > 4 else f"fitted_v7_{arch}.json"))
e = pd.read_csv(src, low_memory=False).drop_duplicates(KEY, keep="last")
e = e[e.origin.isin(CAND) & (e.status == "ok")].reset_index(drop=True)
e["arch_"] = arch
e = M.annotate(e)
e["pred"] = M.predict(M.geometry(e), p)
r = e.loc[e.groupby("case").pred.idxmin(), ["case", "config", "family", "origin", "pred"]]
r["pred_us"] = (r.pred / 1e3).round(2)
r.drop(columns="pred").to_csv(out, index=False)
print(len(r), "picks; same as rules:", int((r.origin == "heuristic").sum()), r.family.value_counts().to_dict())
