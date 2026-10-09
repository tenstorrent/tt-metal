"""Cross-validate model7 (ABLATE=term,... switches terms off): 5-fold by problem per arch, plain argmin picks.
Reports regret and the regression distribution per set, ranking quality, the K-step bias, and every constant: the
all-data fit, its spread over the folds, and whether it sits at a bound. PINNED constants are held at their measured
values. usage: [ABLATE=...] cv7.py [OUT_JSON]"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
import fit

fit.predict, fit.geometry, fit.PARAMS, fit.NAMES = M.predict, M.geometry, M.PARAMS, list(M.PARAMS)
fit.LO.update(M.LO)
fit.HI.update(M.HI)
from data import load, SETS, CAND, evaluate, rank_quality, gm


tag = (os.environ.get("ABLATE", "") or "full") + ("+" + os.environ["EXTRA"] if os.environ.get("EXTRA") else "")
for k in filter(None, os.environ.get("UNPIN", "").split(",")):  # UNPIN=name,...: fit these WH constants freely
    if k in M.PINNED:
        M.PINNED[k] = ({a: v for a, v in M.PINNED[k][0].items() if a != "wh"}, M.PINNED[k][1])
    tag += f"+unpin_{k}"
for kv in filter(
    None, os.environ.get("PIN", "").split(",")
):  # PIN=name=value,...: hold extra WH constants at these values
    k, v = kv.split("=")
    M.PINNED[k] = ({**M.PINNED.get(k, ({}, ""))[0], "wh": float(v)}, "env PIN")
    tag += f"+{k}={v}"
if os.environ.get("ABSW"):  # weight of the absolute log error in the fit (relative error has weight 1)
    fit.ABS_W = float(os.environ["ABSW"])
    tag += f"+absw={fit.ABS_W}"
d = M.annotate(load(list(SETS)))
rows = {a: np.flatnonzero((d.arch_ == a).to_numpy()) for a in ("wh", "bh")}
pred = np.full(len(d), np.nan)
folds, allp = {a: [] for a in rows}, {}
for a, rr in rows.items():
    M.pin(fit, a)
    probs = d.problem_id.iloc[rr].unique()
    np.random.default_rng(0).shuffle(probs)
    for k in range(5):
        test = np.isin(d.problem_id.iloc[rr], probs[k::5])
        p = fit.fit(d, rr[~test])
        folds[a].append(p)
        pred[rr[test]] = M.predict(M.geometry(d.iloc[rr[test]]), p)
    allp[a] = fit.fit(d, rr)
np.save(f"pred_cv_m7_{tag.replace(',', '_')}.npy", pred)
for a in allp:
    json.dump(allp[a], open(f"abl/fitted_{tag.replace(',', '_')}_{a}.json", "w"), indent=1)
    if tag == "full":
        json.dump(allp[a], open(f"fitted_v7_{a}.json", "w"), indent=1)

r = evaluate(d, pred, "", quiet=True)
acc = rank_quality(d, pred)
# K-step bias: predicted minus measured log-time change between adjacent K depths of the same tiling
c = d.assign(pred=pred, kb=d.in0_block_w.fillna(1))
c = c[c.origin.isin(CAND)]
cols = ["problem_id", "family", "grid_x", "grid_y", "per_core_M", "per_core_N", "out_block_h", "out_block_w"]
cols += ["out_subblock_h", "out_subblock_w", "fuse_batch"]
e = []
for _, g in c.groupby(cols, dropna=False):
    g = g.sort_values("kb").drop_duplicates("kb")
    t, q, kb = np.log(g.device_ns.to_numpy()), np.log(g.pred.to_numpy()), g.kb.to_numpy()
    e += [((q[i + 1] - q[i]) - (t[i + 1] - t[i]), kb[i]) for i in range(len(g) - 1)]
e = np.array(e)
kbias = {
    b: round(float(np.median(e[m, 0])), 3)
    for b, m in (("kb1", e[:, 1] == 1), ("kb2-8", (e[:, 1] >= 2) & (e[:, 1] <= 8)), ("kb>8", e[:, 1] > 8))
}
BINS = [1.05, 1.10, 1.25, 1.5, np.inf]
sets = {}
for s, g in r.groupby("set", sort=False):
    reg = g.vs_legacy[g.vs_legacy > 1.05]
    sets[s] = dict(
        regret=round(gm(g.regret), 4),
        p95_regret=round(float(g.regret.quantile(0.95)), 3),
        regr=int(len(reg)),
        bins=pd.cut(reg, BINS).value_counts().sort_index().astype(int).tolist(),
        worst=round(float(g.vs_legacy.max()), 3),
    )
const = {}
for a in rows:
    F = pd.DataFrame(folds[a])
    for k in M.PARAMS:
        lo, hi = M.LO.get(k, 1e-3), M.HI.get(k, 1e7)
        v = allp[a][k]
        const.setdefault(k, {})[a] = dict(
            value=round(float(v), 4),
            fold_spread=round(float(F[k].max() / max(F[k].min(), 1e-12)), 3),
            pinned=a in M.PINNED.get(k, ({},))[0],
            at_bound=bool(a not in M.PINNED.get(k, ({},))[0] and (v <= lo * 1.02 or v >= hi / 1.02)),
        )
out = dict(ablate=tag, sets=sets, pairacc=round(acc[0], 4), med_err=round(acc[1], 4), kbias=kbias, constants=const)
line = f"{tag:10s} " + "  ".join(f"{s} {v['regret']:.3f}/{v['regr']} {v['bins']}" for s, v in sets.items())
print(line + f" | pairacc {acc[0]:.3f} | Kbias {kbias}")
for k, v in const.items():
    print(
        f"   {k:10s} "
        + "  ".join(
            f"{a} {x['value']:<10.4g} x{x['fold_spread']:<6.3g}"
            + (" PIN" if x["pinned"] else "")
            + (" BOUND" if x["at_bound"] else "")
            for a, x in v.items()
        )
    )
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
