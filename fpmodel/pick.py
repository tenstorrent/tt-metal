"""Pick a config per case from an enumerated-candidates CSV (sweep_enumerated.py --enumerate-only), with the frozen constants
in fitted.json and the 5% shallow-K tie-break. Writes a picks CSV for time_picks.py: case,config,family,origin,pred_us.
usage: pick.py ENUM_CSV ARCH OUT_CSV"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, numpy as np, pandas as pd

sys.path.insert(0, ".")
from data import KEY, CAND
from model import geometry, predict

MARGIN = 0.05
src, arch, out = sys.argv[1:4]
d = pd.read_csv(src, low_memory=False).drop_duplicates(KEY, keep="last")
d = d[d.origin.isin(CAND) & (d.status == "ok")].reset_index(drop=True)
d["arch_"] = arch
d["pred"] = predict(geometry(d), json.load(open("fitted.json"))[arch])
rows = []
for case, g in d.groupby("case", sort=False):
    best = g.pred.min()
    near = g[g.pred <= best * (1 + MARGIN)]
    p = near.assign(kb=near.in0_block_w.fillna(1)).sort_values(["kb", "pred"]).iloc[0]
    rows.append(dict(case=case, config=p.config, family=p.family, origin=p.origin, pred_us=round(p.pred / 1e3, 2)))
r = pd.DataFrame(rows)
r.to_csv(out, index=False)
print(
    f"{len(r)} picks; same as the rules' pick: {(r.origin == 'heuristic').sum()}; by family: {r.family.value_counts().to_dict()}"
)
