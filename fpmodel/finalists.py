"""Finalist timing: per problem, the model's top K candidates predicted within MARGIN of its best. Writes the cases with
more than one finalist and a pre-filled sweep output marking every other enumerated config done, so
sweep_enumerated.py --resume times only the finalists (plus legacy and the rules' pick, which it always times).
usage: finalists.py CASES_CSV ENUM_CSV OUT_DIR CONST_JSON RESID_LGB [K=3] [MARGIN=1.2]"""
import sys, os, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R

cases_csv, enum_csv, out_dir, cpath, lpath = sys.argv[1:6]
K = int(sys.argv[6]) if len(sys.argv) > 6 else 3
MARGIN = float(sys.argv[7]) if len(sys.argv) > 7 else 1.2
p = json.load(open(cpath))
raw = pd.read_csv(enum_csv, low_memory=False)
e = raw[(raw.origin == "enumerated") & (raw.status == "ok")].reset_index(drop=True).assign(arch_="wh")
e = M.annotate(e)
b, g, pr = R.base_parts(e, p)
e["pred"] = b * np.exp(lgb.Booster(model_file=lpath).predict(R.features(e, g, pr)))
e["rk"] = e.groupby("case").pred.rank(method="first")
e["rel"] = e.pred / e.groupby("case").pred.transform("min")
fin = e[(e.rk <= K) & (e.rel <= MARGIN)]
n = fin.groupby("case").size()
multi = set(n.index[n > 1])
keys = set(zip(fin.problem_id, fin.config))
os.makedirs(out_dir, exist_ok=True)
c = pd.read_csv(cases_csv, low_memory=False)
c[c.case.isin(multi)].to_csv(f"{out_dir}/cases.csv", index=False)
skip = raw[
    raw.case.isin(multi)
    & (raw.origin == "enumerated")
    & ~pd.Series(list(zip(raw.problem_id, raw.config))).isin(keys).to_numpy()
].assign(status="skip", device_ns=np.nan)
skip.to_csv(f"{out_dir}/sweep.csv", index=False)
fin[["case", "problem_id", "config", "pred", "rk"]].to_csv(f"{out_dir}/finalists.csv", index=False)
print(
    f"{e.case.nunique()} problems; {len(multi)} with >1 finalist; finalists to time {int(n[n > 1].sum())}; mean finalists {n.mean():.2f}"
)
