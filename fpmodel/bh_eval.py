"""Out-of-sample BH: score frozen model7 BH constants on the designed BH problems not used in training.
usage: bh_eval.py [CONST_JSON]"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import data, model7 as M
from data import evaluate, rank_quality, gm

W = "/localdev/rmiller/mm-oob-model-work"
data.SETS["bh_all"] = (f"{W}/frozen/bh_designed_all_1296.csv", "bh")
train = set(pd.read_csv(data.SETS["bh_designed"][0], usecols=["problem_id"]).problem_id.astype(str))
d = data.load(["bh_all"])
raw = d.problem_id.str.split(":", n=1).str[1]
d = d[~raw.isin(train)].reset_index(drop=True)
d = M.annotate(d)
p = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "fitted_v7_bh.json"))
pred = M.predict(M.geometry(d), p)
r = evaluate(d, pred, "v7 frozen, unseen BH")
acc = rank_quality(d, pred)
err = np.log(pred / d.device_ns.to_numpy())
rel = err - pd.Series(err).groupby(d.problem_id.to_numpy()).transform("median").to_numpy()
v = r.vs_legacy.dropna()
rv = r.rules_vs_legacy.dropna()
BINS = [1.05, 1.10, 1.25, 1.5, np.inf]
print(f"problems {len(r)}  pairacc {acc[0]:.3f}  median |within-problem err| {np.median(np.abs(rel)):.3f}")
for name, x, rg in (("model", v, r.regret), ("rules", rv, r.rules_regret.dropna())):
    reg = x[x > 1.05]
    print(
        f"  {name}: regret {gm(rg):.3f} p95 {rg.quantile(.95):.2f} | vs legacy gm {gm(x):.3f} p50 {x.median():.3f} p95 {x.quantile(.95):.3f}"
        f" worst {x.max():.2f} | >1.05 {len(reg)} ({len(reg)/len(x):.1%}) bins {pd.cut(reg, BINS).value_counts().sort_index().tolist()}"
    )
print(f"  best available vs legacy gm {gm((r.best / r.legacy).dropna()):.3f}")
r.to_csv(f"{W}/miss/bh_unseen_v7.csv", index=False)
