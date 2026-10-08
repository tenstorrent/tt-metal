import sys, os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
pred = np.load(sys.argv[1] if len(sys.argv) > 1 else "pred_cv.npy")
d["pred"] = pred
rows = []
for p, g in d.groupby("problem_id", sort=False):
    c = g[g.origin.isin(CAND)]
    pk = c.loc[c.pred.idxmin()]
    b = c.loc[c.device_ns.idxmin()]
    leg = g[g.origin == "legacy"].device_ns.min()
    tk = lambda r: (
        r.family,
        r.grid_x,
        r.grid_y,
        r.per_core_M,
        r.per_core_N,
        r.out_block_h,
        r.out_block_w,
        r.fuse_batch,
    )
    why = (
        "family"
        if pk.family != b.family
        else "tiling"
        if tk(pk) != tk(b)
        else "K"
        if pk.in0_block_w != b.in0_block_w
        else "subblock"
        if (pk.out_subblock_h, pk.out_subblock_w) != (b.out_subblock_h, b.out_subblock_w)
        else "same"
    )
    kdir = np.sign(pk.in0_block_w - b.in0_block_w)
    rows.append(
        dict(
            p=p,
            set=pk.set,
            regret=pk.device_ns / b.device_ns,
            vl=pk.device_ns / leg,
            why=why,
            kdir=kdir,
            pfam=pk.family,
            bfam=b.family,
            us=b.device_ns / 1e3,
            pk=pk.in0_block_w,
            bk=b.in0_block_w,
            Kt=np.ceil(pk.K / 32),
            pred_ratio=pk.pred / c.loc[c.device_ns.idxmin()].pred,
        )
    )
r = pd.DataFrame(rows)
bad = r[r.vl > 1.05]
print("regressions vs legacy by cause:\n", bad.groupby(["set", "why"]).size().unstack(fill_value=0))
print(
    "all picks with regret > 1.10 by cause:\n", r[r.regret > 1.1].groupby(["set", "why"]).size().unstack(fill_value=0)
)
print("K misses: too deep vs too shallow:", r[r.why == "K"].kdir.value_counts().to_dict())
print("family misses (picked -> best):\n", r[r.why == "family"].groupby(["pfam", "bfam"]).size().sort_values().tail(8))
r.to_pickle("misses.pkl")
print(
    r.sort_values("regret").tail(12)[["p", "regret", "vl", "why", "pfam", "bfam", "pk", "bk", "Kt", "us"]].to_string()
)
