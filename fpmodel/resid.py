"""Residual map for model7: where is the prediction systematically wrong?

Rows are timed configs from the training sets (out-of-fold CV predictions) and from fully swept device sets predicted
with frozen constants (out of sample). For each row, err = log(pred / measured); rel = err minus its problem's median
(what selection sees: a regime predicted too fast relative to the other configs of the same problem gets over-picked).
Cells of each regime dimension (and some pairs) are ranked by |median rel| and |median err|.
usage: resid.py OUT_JSON  (needs pred_cv_m7_full.npy and fitted_v7_<arch>.json)"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import model7 as M
from data import load, SETS, CAND

W = "/localdev/rmiller/mm-oob-model-work"
SWEPT = [(f"{W}/fresh/fresh_sweep.csv", "fresh90001 swept"), (f"{W}/miss/miss_sweep.csv", "v6 misses swept")]
MIN_N = 30


def regimes(x, g, parts):
    cut = lambda v, b, l: pd.cut(np.asarray(v, float), b, labels=l).astype(str)
    srcname = np.array(["dram", "L1", "sharded"])
    reads = g["obh"] * g["kb"] + g["kb"] * g["obw"]
    nK, read, comp = parts["nK"], parts["read"], parts["comp"]
    return pd.DataFrame(
        dict(
            family=x.family.to_numpy(),
            cores=cut(g["cores"], [0, 8, 32, 63, np.inf], ["≤8", "9-32", "33-63", "64+"]),
            in0=srcname[g["src_a"]],
            in1=srcname[g["src_b"]],
            out=srcname[g["dst_o"]],
            nK=cut(nK, [0, 1, 4, 16, np.inf], ["1", "2-4", "5-16", ">16"]),
            kb=cut(g["kb"], [0, 1, 2, 8, np.inf], ["1", "2", "3-8", ">8"]),
            Mt=cut(g["Mt"], [0, 1, 8, 64, np.inf], ["1", "2-8", "9-64", ">64"]),
            tiles_per_step=cut(reads, [0, 8, 32, 128, np.inf], ["≤8", "9-32", "33-128", ">128"]),
            blocks=cut(g["nob"], [0, 1, 8, np.inf], ["1", "2-8", ">8"]),
            bound=np.where(read >= comp, "read-bound", "compute-bound"),
            dtype=(x.a_dtype.astype(str) + "×" + x.b_dtype.astype(str)).to_numpy(),
            fidelity=x.fidelity.astype(str).to_numpy(),
            spill=np.where(g["reloads"] > 0, np.where(g["tb_p"] >= 4096, "fp32 reload", "reload"), "none/L1acc"),
            batch=cut(g["B"], [0, 1, 64, np.inf], ["1", "2-64", ">64"]),
            fused=np.where(x.fuse_batch.fillna(0).to_numpy() == 1, "fused", "looped"),
            ragged=np.where(
                (x.M.to_numpy() % 32 != 0) | (x.K.to_numpy() % 32 != 0) | (x.N.to_numpy() % 32 != 0),
                "ragged",
                "aligned",
            ),
            size=cut(
                x.device_ns.to_numpy() / 1e3, [0, 20, 200, 2000, np.inf], ["<20us", "20-200us", "0.2-2ms", ">2ms"]
            ),
        ),
        index=x.index,
    )


if __name__ == "__main__":
    parts_all = []
    # training sets, out of fold
    d = M.annotate(load(list(SETS)))
    d["pred"] = np.load("pred_cv_m7_full.npy")
    d = d[d.origin.isin(CAND) | (d.origin == "legacy")]
    d["source"] = "CV " + d["set"]
    d["key"] = d.problem_id
    for a in ("wh", "bh"):
        x = d[d.arch_ == a]
        _, pr = M.predict(M.geometry(x), json.load(open(f"fitted_v7_{a}.json")), parts=True)
        parts_all.append((x, M.geometry(x), pr))
    # swept device sets, frozen constants
    for path, name in SWEPT:
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, low_memory=False)
        s = s[(s.status == "ok") & s.origin.isin(list(CAND) + ["legacy"])].drop_duplicates(
            ["case", "config"], keep="last"
        )
        s = s.reset_index(drop=True).assign(arch_="wh", source=name, key=lambda z: z.case)
        s = M.annotate(s)
        p = json.load(open("fitted_v7_wh.json"))
        s["pred"], pr = M.predict(M.geometry(s), p, parts=True)
        parts_all.append((s, M.geometry(s), pr))
    rows = []
    for x, g, pr in parts_all:
        pr = {k: (np.asarray(v) * np.ones(len(x))) for k, v in pr.items()}
        r = regimes(x, g, pr)
        r["source"] = x.source.to_numpy()
        r["key"] = x.key.astype(str).to_numpy()
        r["err"] = np.log(x.pred.to_numpy() / x.device_ns.to_numpy())
        rows.append(r.reset_index(drop=True))
    R = pd.concat(rows, ignore_index=True)
    R["rel"] = R.err - R.groupby(["source", "key"]).err.transform("median")
    DIMS = [c for c in R.columns if c not in ("source", "key", "err", "rel")]

    def cell(g):
        return dict(
            n=int(len(g)),
            problems=int(g.key.nunique()),
            err=round(float(g.err.median()), 3),
            rel=round(float(g.rel.median()), 3),
            abs_rel=round(float(g.rel.abs().median()), 3),
        )

    out = dict(n=int(len(R)), sources=R.source.value_counts().to_dict(), dims={}, pairs={}, overall={})
    for src, g in R.groupby("source"):
        out["overall"][src] = cell(g)
    for dim in DIMS:
        out["dims"][dim] = {str(k): cell(g) for k, g in R.groupby(dim) if len(g) >= MIN_N}
    PAIRS = [
        ("family", "cores"),
        ("family", "in0"),
        ("family", "nK"),
        ("in0", "tiles_per_step"),
        ("family", "Mt"),
        ("bound", "family"),
    ]
    for a, b in PAIRS:
        out["pairs"][f"{a}|{b}"] = {f"{k[0]}|{k[1]}": cell(g) for k, g in R.groupby([a, b]) if len(g) >= MIN_N}
    json.dump(out, open(sys.argv[1], "w"), indent=1)

    print("rows", len(R), out["sources"])
    print("overall (median err, median rel, median |rel|):")
    for k, v in out["overall"].items():
        print(f"  {k:22s} n={v['n']:6d}  err {v['err']:+.3f}  rel {v['rel']:+.3f}  |rel| {v['abs_rel']:.3f}")
    flat = [(f"{dim}={k}", v) for dim, cells in out["dims"].items() for k, v in cells.items()]
    flat += [(f"{p}={k}", v) for p, cells in out["pairs"].items() for k, v in cells.items()]
    print(
        "\nlargest systematic within-problem errors (rel < 0: predicted too fast vs the problem's other configs -> over-picked):"
    )
    for name, v in sorted(flat, key=lambda t: -abs(t[1]["rel"]))[:30]:
        print(
            f"  {name:42s} n={v['n']:6d} probs={v['problems']:4d}  rel {v['rel']:+.3f}  err {v['err']:+.3f}  |rel| {v['abs_rel']:.3f}"
        )
