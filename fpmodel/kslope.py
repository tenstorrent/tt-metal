"""Per-K-step cost, measured vs model: within a tiling (all but in0_block_w), slope of time over K steps per core."""
import sys, os, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
F = json.load(open("fitted.json"))
d = d[d.origin.isin(CAND) & (d.family != "multicore")].reset_index(drop=True)
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
cols = {}
for a in ("wh", "bh"):
    m = (d.arch_ == a).to_numpy()
    g = geometry(d[m])
    t, pr = predict(g, F[a], parts=True)
    for k, v in dict(
        pred=t, steps=g["nob"] * g["nK"], rx0=g["rx0"], rx1=g["rx1"], comp=pr["comp"], read=pr["read"], nK=g["nK"]
    ).items():
        cols.setdefault(k, np.zeros(len(d)))[m] = v
for k, v in cols.items():
    d[k] = v
rows = []
for k, g in d.groupby("tk"):
    if g.in0_block_w.nunique() < 3:
        continue
    A = np.column_stack([np.ones(len(g)), g.steps])
    sr = np.linalg.lstsq(A, g.device_ns.to_numpy(), rcond=None)[0][1]
    sp = np.linalg.lstsq(A, g.pred.to_numpy(), rcond=None)[0][1]
    r = g.iloc[0]
    rows.append(
        dict(
            arch=r.arch_,
            fam=r.family,
            rx=max(r.rx0, r.rx1),
            src=r.a_mem[:4] + "/" + r.b_mem[:4],
            real=sr,
            pred=sp,
            kbmax=g.in0_block_w.max(),
            kbmin=g.in0_block_w.min(),
            wt=r.out_block_h * r.out_block_w,
            acc32=r.fp32_acc,
            l1acc=r.packer_l1_acc,
            a_mem=r.a_mem,
            b_mem=r.b_mem,
            nsteps_max=g.steps.max(),
        )
    )
s = pd.DataFrame(rows)
s["rxb"] = pd.cut(s.rx, [-1, 0, 7, 15, 31, 63, 200])
print("per-step cost (ns; slope of time over K steps per core), median, by family x receivers")
print(
    s.groupby(["arch", "fam", "rxb"], observed=True)
    .agg(n=("real", "size"), real=("real", "median"), model=("pred", "median"))
    .round(0)
    .to_string()
)
s["wtb"] = pd.cut(s.wt, [0, 1, 4, 16, 64, 1e9])
print(
    s.groupby(["arch", "wtb"], observed=True)
    .agg(n=("real", "size"), real=("real", "median"), model=("pred", "median"))
    .round(0)
    .to_string()
)
s.to_pickle("kslope.pkl")
