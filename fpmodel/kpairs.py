"""Same tiling (everything but in0_block_w), two K depths: measured vs predicted ratio deep/shallow."""
import sys, os, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
pred = np.load("pred_cv.npy")
d["pred"] = pred
d = d[d.origin.isin(CAND)]
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
g_all = {}
for a in ("wh", "bh"):
    m = d.arch_ == a
    g = geometry(d[m])
    g_all[a] = (d.index[m], g)
d["nK"] = np.nan
d["dbuf"] = np.nan
for a, (idx, g) in g_all.items():
    d.loc[idx, "nK"] = g["nK"]
    d.loc[idx, "dbuf"] = g["dbuf"]
rows = []
for k, g in d.groupby("tk"):
    if g.in0_block_w.nunique() < 2:
        continue
    g = g.sort_values("in0_block_w")
    s, dp = g.iloc[0], g.iloc[-1]
    rows.append(
        dict(
            pid=s.problem_id,
            arch=s.arch_,
            fam=s.family,
            kb_s=s.in0_block_w,
            kb_d=dp.in0_block_w,
            nK_d=dp.nK,
            real=dp.device_ns / s.device_ns,
            pred=dp.pred / s.pred,
            us=s.device_ns / 1e3,
            src=s.a_mem[:4] + "/" + s.b_mem[:4],
            nob=s.per_core_M / s.out_block_h * s.per_core_N / s.out_block_w,
            l1acc=s.packer_l1_acc,
            acc32=s.fp32_acc,
        )
    )
r = pd.DataFrame(rows)
r["err"] = np.log(r.pred / r.real)
r["wrong"] = ((r.real > 1.05) & (r.pred < 1)) | ((r.real < 1 / 1.05) & (r.pred > 1))
print(
    "pairs",
    len(r),
    "order wrong (>5% real gap):",
    int(r.wrong.sum()),
    " deep really slower >5%:",
    int((r.real > 1.05).sum()),
    " of those predicted faster:",
    int(((r.real > 1.05) & (r.pred < 1)).sum()),
)
for k in ["fam", "src", "arch", "l1acc", "acc32"]:
    print(
        r.groupby(k).agg(n=("err", "size"), wrong=("wrong", "sum"), med_err=("err", "median")).round(2).T.to_string(),
        "\n",
    )
r["nKd"] = pd.cut(r.nK_d, [0, 1, 2, 4, 1e9])
r["nobb"] = pd.cut(r.nob, [0, 1, 4, 1e9])
r["usb"] = pd.cut(r.us, [0, 50, 200, 1000, 1e9])
for k in ["nKd", "nobb", "usb"]:
    print(
        r.groupby(k, observed=True)
        .agg(n=("err", "size"), wrong=("wrong", "sum"), med_err=("err", "median"))
        .round(2)
        .T.to_string(),
        "\n",
    )
r.to_pickle("kpairs.pkl")
