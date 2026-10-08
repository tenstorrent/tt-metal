"""Which mechanism explains the K-pair errors? Pair = same tiling, shallowest vs deepest K; err = log(pred ratio / real ratio).
err < 0: model favours the deep config more than it should."""
import sys, os, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *
from model import *
from sklearn.tree import DecisionTreeRegressor, export_text

d = load(list(SETS))
d["pred"] = np.load("pred_cv.npy")
d = d[d.origin.isin(CAND) & (d.family != "multicore")].reset_index(drop=True)
F = json.load(open("fitted.json"))
feat = {}
for a in ("wh", "bh"):
    m = (d.arch_ == a).to_numpy()
    g = geometry(d[m])
    t, pr = predict(g, F[a], parts=True)
    b0 = g["obh"] * g["kb"] * g["tb_a"]
    b1 = g["kb"] * g["obw"] * g["tb_b"]
    cb = np.where(g["dbuf"], 2, 1) * (b0 + b1) + g["obh"] * g["obw"] * np.maximum(g["tb_o"], g["tb_p"])
    vals = dict(
        kb=g["kb"],
        nK=g["nK"],
        b0_KB=b0 / 1e3,
        b1_KB=b1 / 1e3,
        cb_KB=cb / 1e3,
        dbuf=g["dbuf"].astype(float),
        rx0=g["rx0"],
        rx1=g["rx1"],
        rd0=g["rd0"],
        rd1=g["rd1"],
        read_over_comp=np.log(pr["read"] / pr["comp"]),
        nob=g["nob"],
        ph=g["ph"],
        nsb=g["nsb"],
        src_a=g["src_a"].astype(float),
        src_b=g["src_b"].astype(float),
        reloads=g["reloads"],
    )
    for k, v in vals.items():
        feat.setdefault(k, np.zeros(len(d)))[m] = v
for k, v in feat.items():
    d[k] = v
T = [
    "problem_id",
    "family",
    "grid_x",
    "grid_y",
    "per_core_M",
    "per_core_N",
    "out_block_h",
    "out_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "fuse_batch",
    "transpose_mcast",
]
d["tk"] = d[T].astype(str).agg("|".join, axis=1)
rows = []
for k, g in d.groupby("tk"):
    if g.in0_block_w.nunique() < 2:
        continue
    g = g.sort_values("in0_block_w")
    for i in range(1, len(g)):  # every deeper K vs the shallowest
        s, dp = g.iloc[0], g.iloc[i]
        if dp.in0_block_w == s.in0_block_w:
            continue
        r = dict(arch=s.arch_, fam=s.family, real=np.log(dp.device_ns / s.device_ns), pred=np.log(dp.pred / s.pred))
        for f in feat:
            r["d_" + f] = dp[f]  # deep config's quantities
        r["s_read_over_comp"] = s.read_over_comp
        r["s_nK"] = s.nK
        rows.append(r)
p = pd.DataFrame(rows)
p["err"] = p.pred - p.real
p["fam_c"] = p.fam.map({"1d_in0": 0, "1d_in1": 1, "2d": 2, "reuse": 3})
p["bh"] = (p.arch == "bh").astype(float)
X = p[[c for c in p.columns if c.startswith(("d_", "s_"))] + ["fam_c", "bh"]]
print("pairs", len(p), " mean err", p.err.mean().round(3), " |err|>0.1:", int((p.err.abs() > 0.1).sum()))
t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60).fit(X, p.err)
print(export_text(t, feature_names=list(X.columns), decimals=2))
p.to_pickle("kresid.pkl")
