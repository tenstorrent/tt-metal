"""Coverage matrix and failure taxonomy for a model version.

Every evaluated problem is placed in regime cells (memory source, spill path, M/K/N size, batch, dtype, fidelity, epilogue,
raggedness, grid, arch). For each cell: how many problems the version has been judged on (CV on the training sets, fresh
device sets) and how it did there (regressions vs legacy, severe ones, worst). Every regression is tagged with what the pick
changed relative to legacy's config (family, tiling, K depth, subblock). usage: coverage.py OUT_JSON"""
import os, sys, json, math

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data import load, SETS, CAND

W = "/localdev/rmiller/mm-oob-model-work"
V6_CV = "pred_cv_x_noc_rl_burstl1+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100.npy"
FRESH = [(f"{W}/fresh/fresh_timed.csv", f"{W}/fresh/fresh_cases.csv", "fresh90001")]
THIN = 15


def regimes(x):
    """problem-level regime labels from case attributes (one row per problem)"""
    t = lambda v: np.ceil(np.asarray(v, float) / 32)
    Mt, Kt, Nt = t(x.M), t(x.K), t(x.N)
    mem = lambda m: np.where(m == "dram", "dram", np.where(m == "l1", "l1", "sharded"))
    am, bm = mem(x.a_mem.astype(str).to_numpy()), mem(x.b_mem.astype(str).to_numpy())
    src = np.where(
        (am == "sharded") | (bm == "sharded"),
        "sharded input",
        np.where((am == "l1") | (bm == "l1"), "interleaved-L1 input", "DRAM inputs"),
    )
    acc, l1 = x.fp32_acc.fillna(0).to_numpy() == 1, x.packer_l1_acc.fillna(0).to_numpy() == 1
    spill = np.where(l1, "L1 acc", np.where(acc, "fp32 partials, no L1 acc", "no L1 acc"))
    act = x.activation.fillna("").astype(str).replace("nan", "").to_numpy() != ""
    bias = x.bias.fillna(0).to_numpy() == 1
    epi = np.where(bias & act, "bias + act", np.where(bias, "bias", np.where(act, "activation", "none")))
    cut = lambda v, b, l: pd.cut(v, b, labels=l).astype(str)
    B = x.batch.to_numpy(float)
    rag = (np.asarray(x.M) % 32 != 0) | (np.asarray(x.K) % 32 != 0) | (np.asarray(x.N) % 32 != 0)
    grid = x.core_grid.fillna("").astype(str).replace("nan", "").to_numpy() if "core_grid" in x else np.full(len(x), "")
    return pd.DataFrame(
        dict(
            arch=x.arch_.to_numpy(),
            source=src,
            spill=spill,
            epilogue=epi,
            M=cut(Mt, [0, 1, 4, 32, 256, np.inf], ["Mt 1", "Mt 2-4", "Mt 5-32", "Mt 33-256", "Mt >256"]),
            K=cut(Kt, [0, 4, 32, 256, np.inf], ["Kt ≤4", "Kt 5-32", "Kt 33-256", "Kt >256"]),
            N=cut(Nt, [0, 4, 32, 256, np.inf], ["Nt ≤4", "Nt 5-32", "Nt 33-256", "Nt >256"]),
            batch=cut(B, [0, 1, 64, np.inf], ["batch 1", "batch 2-64", "batch >64"]),
            dtype=(x.a_dtype.astype(str) + " × " + x.b_dtype.astype(str)).to_numpy(),
            fidelity=x.fidelity.astype(str).to_numpy(),
            ragged=np.where(rag, "ragged", "tile-aligned"),
            grid=np.where(grid == "", "default grid", "custom grid"),
        ),
        index=x.index,
    )


def cause_vs_legacy(p, l, cand_families=None):
    """what the pick changed relative to legacy's config"""
    if cand_families is not None and l.family not in cand_families:
        return f"candidate gap ({l.family} not enumerated)"
    if p.family != l.family:
        return f"family ({l.family}→{p.family})"
    tk = lambda r: tuple(
        str(r.get(k))
        for k in ("grid_x", "grid_y", "per_core_M", "per_core_N", "out_block_h", "out_block_w", "fuse_batch")
    )
    if tk(p) != tk(l):
        return "tiling"
    if p.in0_block_w != l.in0_block_w:
        return "K deeper" if p.in0_block_w > l.in0_block_w else "K shallower"
    return "subblock/same"


rows = []
# ---- CV on the training sets ----
d = load(list(SETS))
d["pred"] = np.load(V6_CV)
for pid, g in d.groupby("problem_id", sort=False):
    c = g[g.origin.isin(CAND)]
    leg = g[g.origin == "legacy"]
    if not len(leg):
        continue
    pk, lg = c.loc[c.pred.idxmin()], leg.loc[leg.device_ns.idxmin()]
    ratio = pk.device_ns / lg.device_ns
    rows.append(
        dict(
            src="CV " + g.set.iloc[0],
            kind="cv",
            key=pid,
            ratio=ratio,
            cause=cause_vs_legacy(pk, lg, set(c.family)) if ratio > 1.05 else "",
            idx=g.index[0],
        )
    )
cv = pd.DataFrame(rows)
reg_cv = regimes(d.loc[cv.idx].reset_index(drop=True))
cv = pd.concat([cv.reset_index(drop=True), reg_cv], axis=1)
# ---- fresh device sets ----
parts = [cv]
for timed, cases, name in FRESH:
    t = pd.read_csv(timed, low_memory=False)
    cs = pd.read_csv(cases).set_index("case")
    en = pd.read_csv(timed.replace("_timed.csv", "_enum.csv"), low_memory=False)
    fams = en[en.origin.isin(CAND) & (en.status == "ok")].groupby("case").family.agg(set)
    ok = t[t.status == "ok"]
    ns = ok.pivot_table(index="case", columns="origin", values="device_ns", aggfunc="last").dropna(
        subset=["legacy", "model"]
    )
    last = ok.drop_duplicates(["case", "origin"], keep="last").set_index(["case", "origin"])
    fr = []
    for c in ns.index:
        ratio = ns.loc[c, "model"] / ns.loc[c, "legacy"]
        cause = cause_vs_legacy(last.loc[(c, "model")], last.loc[(c, "legacy")], fams.get(c)) if ratio > 1.05 else ""
        fr.append(dict(src=name, kind="device", key=c, ratio=ratio, cause=cause))
    fr = pd.DataFrame(fr)
    attrs = cs.loc[fr.key].reset_index().assign(arch_="wh")
    parts.append(pd.concat([fr.reset_index(drop=True), regimes(attrs).reset_index(drop=True)], axis=1))
A = pd.concat(parts, ignore_index=True)
DIMS = ["arch", "source", "spill", "epilogue", "M", "K", "N", "batch", "dtype", "fidelity", "ragged", "grid"]


def stats(g):
    r = g.ratio
    return dict(
        n=int(len(g)),
        n_cv=int((g.kind == "cv").sum()),
        n_dev=int((g.kind == "device").sum()),
        regr=int((r > 1.05).sum()),
        severe=int((r > 1.25).sum()),
        worst=round(float(r.max()), 3),
        rate=round(float((r > 1.05).mean()), 4),
    )


marg = {dim: {str(k): stats(g) for k, g in A.groupby(dim)} for dim in DIMS}
PAIRS = [("source", "spill"), ("K", "spill"), ("M", "K"), ("source", "M"), ("dtype", "fidelity"), ("batch", "M")]
mats = {}
for a, b in PAIRS:
    mats[f"{a}|{b}"] = dict(
        rows=sorted(A[a].unique().tolist()),
        cols=sorted(A[b].unique().tolist()),
        cells={f"{k[0]}|{k[1]}": stats(g) for k, g in A.groupby([a, b])},
    )
reg = A[A.ratio > 1.05].sort_values("ratio", ascending=False)
tax = (
    reg.groupby("cause")
    .agg(n=("ratio", "size"), severe=("ratio", lambda r: int((r > 1.25).sum())), worst=("ratio", "max"))
    .reset_index()
)
out = dict(
    version="v6",
    thin=THIN,
    totals=dict(n=int(len(A)), cv=int((A.kind == "cv").sum()), device=int((A.kind == "device").sum())),
    marginals=marg,
    matrices=mats,
    taxonomy=[
        dict(cause=r.cause, n=int(r.n), severe=int(r.severe), worst=round(float(r.worst), 3)) for r in tax.itertuples()
    ],
    regressions=[
        dict(
            src=r.src,
            key=str(r.key),
            ratio=round(float(r.ratio), 3),
            cause=r.cause,
            **{k: str(getattr(r, k)) for k in DIMS},
        )
        for r in reg.itertuples()
    ],
)
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print("problems:", out["totals"], " regressions:", len(reg), " severe:", int((reg.ratio > 1.25).sum()))
print(tax.sort_values("n", ascending=False).to_string(index=False))
for dim in DIMS:
    thin = [k for k, s in marg[dim].items() if s["n"] < THIN]
    print(
        f"{dim:9s}",
        "  ".join(f"{k}: n={s['n']} r={s['regr']} sev={s['severe']}" for k, s in marg[dim].items()),
        ("  THIN: " + ", ".join(thin)) if thin else "",
    )
