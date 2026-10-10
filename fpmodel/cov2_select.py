"""Coverage round 2: sibling sweeps in the regimes where v18r2's suite picks still regress.
Regimes (from the pool of suite-shape neighbours, problems not timed in round 1):
  lowM     per-core M of a few tiles, deep K, B in DRAM  (K block depth)
  big      M, N >= 2048, K >= 1024                        (subblock shape, K block depth)
  batched  batch >= 16, tiny M and K                      (K block depth, subblock shape)
For each problem: v18r2's top 4, plus every sibling of its top 2 (same family, grid, per-core and output block; any K
block depth and subblock), plus 2 random configs. Writes cases and a pre-filled sweep output (other configs 'skip') so
sweep_enumerated.py --resume times only those.
usage: cov2_select.py POOL_CASES POOL_ENUM ROUND1_CASES OUT_DIR N_LOWM N_BIG N_BATCHED"""
import sys, json, os

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R

cases_csv, enum_csv, r1_csv, out_dir = sys.argv[1:5]
quota = dict(lowM=int(sys.argv[5]), big=int(sys.argv[6]), batched=int(sys.argv[7]))
W = "/localdev/rmiller/mm-oob-model-work"
p = json.load(open(f"{W}/fresh/frozen_v18r2_wh.json"))
bst = lgb.Booster(model_file=f"{W}/fresh/frozen_v18r2_wh.lgb")
rng = np.random.default_rng(109001)

c = pd.read_csv(cases_csv, low_memory=False)
c = c[~c.case.isin(set(pd.read_csv(r1_csv).case))]
Mt, Kt = (c.M + 31) // 32, c.K // 32
regime = {
    "lowM": (Mt <= 8) & (Kt >= 32) & (c.b_mem == "dram"),
    "big": (c.M >= 2048) & (c.N >= 2048) & (c.K >= 1024),
    "batched": (c.batch >= 16) & (c.M <= 256) & (c.K <= 256),
}
sel = {}
for k, m in regime.items():
    pool = sorted(set(c.case[m]) - set(sel))
    for x in rng.choice(pool, min(quota[k], len(pool)), replace=False):
        sel[x] = k

raw = pd.read_csv(enum_csv, low_memory=False)
e = (
    raw[(raw.origin == "enumerated") & (raw.status == "ok") & raw.case.isin(sel)]
    .reset_index(drop=True)
    .assign(arch_="wh")
)
e = M.annotate(e)
b, g, pr = R.base_parts(e, p)
e["pred"] = b * np.exp(bst.predict(R.features(e, g, pr)))
e["rk"] = e.groupby("case").pred.rank(method="first")
sib = [
    "case",
    "family",
    "grid_x",
    "grid_y",
    "per_core_M",
    "per_core_N",
    "out_block_h",
    "out_block_w",
    "fuse_batch",
    "mcast_in0",
    "transpose_mcast",
]
key = lambda z: list(zip(*[z[k].astype(str) for k in sib]))
top2 = set(key(e[e.rk <= 2]))
keep = (e.rk <= 4).to_numpy() | np.array([k in top2 for k in key(e)])
for _, q in e.groupby("case"):
    rest = q.index[~keep[q.index]]
    keep[rng.choice(rest, min(2, len(rest)), replace=False)] = True
tk = set(zip(e.problem_id[keep], e.config[keep]))
os.makedirs(out_dir, exist_ok=True)
pd.read_csv(cases_csv, low_memory=False).pipe(lambda z: z[z.case.isin(sel)]).assign(tier="cov2").to_csv(
    f"{out_dir}/cases.csv", index=False
)
skip = raw[
    raw.case.isin(sel)
    & (raw.origin == "enumerated")
    & ~pd.Series(list(zip(raw.problem_id, raw.config))).isin(tk).to_numpy()
]
skip.assign(status="skip", device_ns=np.nan).to_csv(f"{out_dir}/sweep.csv", index=False)
n = pd.Series(keep).groupby(e.case.map(sel).to_numpy()).agg(["sum", "size"])
print("selected", pd.Series(sel).value_counts().to_dict())
print("configs to time (+legacy, rules) per regime:\n", n)
