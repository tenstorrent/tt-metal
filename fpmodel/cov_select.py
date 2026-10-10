"""Coverage-driven sampling for the residual: from an enumerated pool, choose the problems whose contending configs sit
furthest from the residual's training data, and for each the configs worth timing (the residual model's top N, the
base model's top M, a few random ones; legacy and the rules' pick are timed by the harness anyway). Writes the selected
cases CSV and a pre-filled sweep output marking every other config as done, so sweep_enumerated.py --resume times only
the chosen ones. usage: cov_select.py POOL_CASES POOL_ENUM OUT_DIR N_PROBLEMS [ARCH [MODEL_PREFIX]]
(default: WH, frozen v18r; MODEL_PREFIX names <prefix>.json and <prefix>.lgb)"""
import sys, json, os

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R
from data import load, SETS, CAND
from sklearn.neighbors import NearestNeighbors

cases_csv, enum_csv, out_dir, n_pick = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
arch = sys.argv[5] if len(sys.argv) > 5 else "wh"
TOP_R, TOP_B, N_RAND = 6, 3, 2
W = "/localdev/rmiller/mm-oob-model-work"
prefix = sys.argv[6] if len(sys.argv) > 6 else f"{W}/fresh/frozen_v18r_{arch}"
p = json.load(open(f"{prefix}.json"))
bst = lgb.Booster(model_file=f"{prefix}.lgb")

d = M.annotate(load(list(SETS)))
d = d[(d.arch_ == arch) & d.origin.isin(CAND)].reset_index(drop=True)
b, g, pr = R.base_parts(d, p)
F = R.features(d, g, pr)
cols = [c for c in F.columns if c != "log_base_us"]
T = lambda X: np.sign(X[cols].fillna(-1)) * np.log1p(np.abs(X[cols].fillna(-1)))
A = T(F)
mu, sd = A.mean(), A.std().replace(0, 1)
nn = NearestNeighbors(n_neighbors=10).fit(((A - mu) / sd).to_numpy())

raw = pd.read_csv(enum_csv, low_memory=False)
e = raw[(raw.origin == "enumerated") & (raw.status == "ok")].reset_index(drop=True).assign(arch_=arch)
e = M.annotate(e)
be, ge, pe = R.base_parts(e, p)
Fe = R.features(e, ge, pe)
e["base"] = be
e["v18r"] = be * np.exp(bst.predict(Fe))
e["dist"] = nn.kneighbors(((T(Fe) - mu) / sd).to_numpy())[0].mean(1)
e["rk_r"] = e.groupby("case").v18r.rank(method="first")
e["rk_b"] = e.groupby("case").base.rank(method="first")
top = e[e.rk_r <= 10]
score = top.groupby("case").dist.mean().sort_values(ascending=False)
# the most uncovered problems, spread out: farthest-point sampling over their contenders' mean feature vector
cand_cases = score.index[: int(n_pick * 1.5)]
V = ((T(Fe) - mu) / sd).assign(case=e.case.to_numpy())
V = V[e.rk_r.to_numpy() <= 3].groupby("case").mean().loc[cand_cases].to_numpy()
chosen = [0]
dmin = np.linalg.norm(V - V[0], axis=1)
while len(chosen) < min(n_pick, len(cand_cases)):
    i = int(np.argmax(dmin))
    chosen.append(i)
    dmin = np.minimum(dmin, np.linalg.norm(V - V[i], axis=1))
sel = set(cand_cases[chosen])
rng = np.random.default_rng(108001)
keep = (e.rk_r <= TOP_R) | (e.rk_b <= TOP_B)
for c, gq in e[e.case.isin(sel)].groupby("case"):
    rest = gq.index[~keep[gq.index]]
    keep[rng.choice(rest, min(N_RAND, len(rest)), replace=False)] = True
time_keys = set(zip(e.problem_id[keep], e.config[keep]))
os.makedirs(out_dir, exist_ok=True)
pd.read_csv(cases_csv, low_memory=False).pipe(lambda c: c[c.case.isin(sel)]).to_csv(f"{out_dir}/cases.csv", index=False)
skip = raw[
    raw.case.isin(sel)
    & (raw.origin == "enumerated")
    & ~pd.Series(list(zip(raw.problem_id, raw.config))).isin(time_keys).to_numpy()
].assign(status="skip", device_ns=np.nan)
skip.to_csv(f"{out_dir}/sweep.csv", index=False)
n_time = int(keep[e.case.isin(sel)].sum())
print(
    f"pool {e.case.nunique()} problems; selected {len(sel)}; configs to time {n_time} (+legacy, rules); skipped {len(skip)}"
)
print("selected coverage score: median %.2f (pool median %.2f)" % (score.loc[list(sel)].median(), score.median()))
print("sources:", pd.read_csv(f"{out_dir}/cases.csv").tier.value_counts().to_dict())
