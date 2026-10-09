"""CV a model_x variant (MX env): regret-first report, K-slope bias, key constants."""
import os, sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd, json
import model_x as M
import fit

fit.predict = M.predict
fit.PARAMS = M.PARAMS
fit.NAMES = list(M.PARAMS)
fit.geometry = M.geometry
fit.HI.update(eta_max=0.95, link_eff=1.0, bank_frac=1.0)
fit.LO.update(eta_max=0.3, link_eff=0.05, bank_frac=0.02)
for kv in filter(None, os.environ.get("FIX", "").split(",")):  # pin a constant: FIX=name=value
    k, v = kv.split("=")
    fit.LO[k] = float(v) * 0.999
    fit.HI[k] = float(v) * 1.001
    M.PARAMS[k] = float(v)
from data import *

d = load(list(SETS))
if "noc" in M.MX:
    import nocload

    d["link_bytes"] = nocload.link_bytes(M.geometry(d), d)
if "bank" in M.MX:
    import bankload

    d["bank_a"], d["bank_b"], _ = bankload.banks_touched(M.geometry(d), d)
rows = {a: np.flatnonzero((d.arch_ == a).to_numpy()) for a in ("wh", "bh")}
pred = np.full(len(d), np.nan)
allp = {}
for a, rr in rows.items():
    probs = d.problem_id.iloc[rr].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(probs)
    for k in range(5):
        test = np.isin(d.problem_id.iloc[rr], probs[k::5])
        p = fit.fit(d, rr[~test])
        pred[rr[test]] = M.predict(M.geometry(d.iloc[rr[test]]), p)
    allp[a] = fit.fit(d, rr)
tag = (os.environ.get("MX", "") or "base") + ("+" + os.environ["FIX"] if os.environ.get("FIX") else "")
np.save(f"pred_cv_x_{tag.replace(',', '_')}.npy", pred)
r = evaluate(d, pred, "", quiet=True)
acc = rank_quality(d, pred)
# K slope bias (adjacent K depths within a tiling)
c = d.assign(pred=pred, kb=d.in0_block_w.fillna(1))
c = c[c.origin.isin(CAND)]
cols = [
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
]
e = []
for _, g in c.groupby(cols, dropna=False):
    g = g.sort_values("kb").drop_duplicates("kb")
    if len(g) < 2:
        continue
    t, q = np.log(g.device_ns.to_numpy()), np.log(g.pred.to_numpy())
    kb = g.kb.to_numpy()
    e += [((q[i + 1] - q[i]) - (t[i + 1] - t[i]), kb[i]) for i in range(len(g) - 1)]
e = np.array(e)
bias = {
    b: round(float(np.median(e[m, 0])), 3)
    for b, m in (("kb1", e[:, 1] == 1), ("kb2-8", (e[:, 1] >= 2) & (e[:, 1] <= 8)), ("kb>8", e[:, 1] > 8))
}
line = f"{tag:18s} " + "  ".join(
    f"{s} {gm(g.regret):.3f}/{int((g.vs_legacy > 1.05).sum())}" for s, g in r.groupby("set", sort=False)
)
print(line + f"  | pairacc {acc[0]:.3f} med|err| {acc[1]:.3f} | Kbias {bias}")
keys = ["dram_eff", "launch_us", "lat_mcast0", "ack_rx", "reload", "subblock"] + [
    k for k in M.PARAMS if k not in fit.PARAMS or k in ("rl_init", "u2d", "eta_max", "inflight_KB", "launch_core_ns")
]
print("    WH:", {k: round(allp["wh"][k], 3) for k in dict.fromkeys(keys) if k in allp["wh"]})
print("    BH:", {k: round(allp["bh"][k], 3) for k in dict.fromkeys(keys) if k in allp["bh"]})
