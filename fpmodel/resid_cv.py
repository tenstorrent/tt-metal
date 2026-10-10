"""Nested CV of a learned residual on top of the first-principles model (model7, standard terms).

Per arch, 5 folds by problem. In each fold the base constants are refitted on the training problems only; a shallow
gradient-boosted tree then learns the base's within-problem log error (log meas/pred minus the problem's median) from
the mechanism quantities, on the training problems only, and corrects the held-out predictions:
    final = base * exp(shrink * residual(features))
Reports base vs base+residual per set (regret, regressions vs legacy), ranking quality and the size of the correction.
usage: resid_cv.py [OUT_PREFIX]   env: SHRINK (default 1.0), LEAVES (7), TREES (300), LR (0.03)"""
import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lightgbm as lgb
import model7 as M
import fit

fit.predict, fit.geometry, fit.PARAMS, fit.NAMES = M.predict, M.geometry, M.PARAMS, list(M.PARAMS)
fit.LO.update(M.LO)
fit.HI.update(M.HI)
from data import load, SETS, CAND, evaluate, rank_quality, gm

SHRINK = float(os.environ.get("SHRINK", "1.0"))
PARAMS = dict(
    objective="regression_l1",
    num_leaves=int(os.environ.get("LEAVES", "7")),
    max_depth=int(
        os.environ.get("DEPTH", "-1")
    ),  # 2 = pairwise interactions only: the correction is a sum of 1-D and 2-D shape functions
    learning_rate=float(os.environ.get("LR", "0.03")),
    n_estimators=int(os.environ.get("TREES", "300")),
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)
FAM = {"2d": 0, "1d_in0": 1, "1d_in1": 2, "reuse": 3, "multicore": 4}
FID = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def features(d, g, parts):
    """mechanism quantities the base model already computes (no problem identity, no measured values)"""
    n = len(d)
    a = lambda v: np.asarray(v, float) * np.ones(n)
    read, comp, write = a(parts["read"]), a(parts["comp"]), a(parts["write"])
    nK = a(parts["nK"])
    loop = read + np.maximum(nK - 1, 0) * np.maximum(read, comp) + comp
    tot = loop + write + 1.0
    f = pd.DataFrame(
        dict(
            fam=a(pd.Series(g["fam"]).map(FAM).fillna(5)),
            kb=a(g["kb"]),
            nK=nK,
            log_nK=np.log1p(nK),
            obh=a(g["obh"]),
            obw=a(g["obw"]),
            sbh=a(g["sbh"]),
            sbw=a(g["sbw"]),
            nsb=a(g["nsb"]),
            nob=a(g["nob"]),
            cores=a(g["cores"]),
            rx0=a(g["rx0"]),
            rx1=a(g["rx1"]),
            rd0=a(g["rd0"]),
            rd1=a(g["rd1"]),
            src_a=a(g["src_a"]),
            src_b=a(g["src_b"]),
            dst_o=a(g["dst_o"]),
            tb_a=a(g["tb_a"]),
            tb_b=a(g["tb_b"]),
            tb_p=a(g["tb_p"]),
            ph=a(g["ph"]),
            reloads=a(g["reloads"]),
            dbuf=a(g["dbuf"]),
            kpad=a(g["kpad"]),
            bk_a=a(g["bk_a"]),
            bk_b=a(g["bk_b"]),
            step_kb=a((g["obh"] * g["tb_a"] + g["obw"] * g["tb_b"]) * g["kb"]) / 1024,
            read_frac=read * nK / tot,
            write_frac=write / tot,
            read_over_comp=np.log((read + 1) / (comp + 1)),
            log_base_us=np.log(np.maximum(a(parts["t"]), 1.0) / 1e3),
            batch=a(g["B"]),
            Mt=a(g["Mt"]),
            Kt=a(g["Kt"]),
            Nt=a(g["Nt"]),
        )
    )
    return f


def problem_features(d, f):
    """for the offset stage: the row's mechanism features plus problem attributes the within-problem stage never needs"""
    codes = dict(
        mem={"dram": 0, "l1": 1, "l1_height": 2, "l1_width": 3, "l1_block": 4},
        dt={"bf16": 0, "bfp8": 1, "bfp4": 2, "fp32": 3},
        fid=FID,
    )  # fixed codes, so training and pick time agree; sharded layouts and anything unlisted share the next code
    kind = dict(a_mem="mem", b_mem="mem", out_mem="mem", a_dtype="dt", b_dtype="dt", out_dtype="dt", fidelity="fid")
    cat = lambda c: (
        d[c].map(codes[kind[c]]).fillna(len(codes[kind[c]]) + 1).to_numpy(float) if c in d else np.zeros(len(d))
    )
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0).to_numpy() if c in d else np.zeros(len(d))
    return f.assign(
        a_mem=cat("a_mem"),
        b_mem=cat("b_mem"),
        out_mem=cat("out_mem"),
        a_dt=cat("a_dtype"),
        b_dt=cat("b_dtype"),
        o_dt=cat("out_dtype")
        if "out_dtype" not in d
        else d.out_dtype.fillna(d.a_dtype).map(codes["dt"]).fillna(5).to_numpy(float),
        fid=cat("fidelity"),
        fp32=num("fp32_acc"),
        l1acc=num("packer_l1_acc"),
        bias=num("bias"),
        tb=num("transpose_b"),
    )


def problem_offset(d, f, base, corr, rows_tr, rows_te, w):
    """second stage: the problem-level log error left after the within-problem correction (constant per problem, so
    it never changes a pick). Trained on each training problem's median, predicted per row and averaged per problem."""
    e = np.log(d.device_ns.to_numpy()[rows_tr] / base[rows_tr]) - corr[rows_tr]
    pid = d.problem_id.to_numpy()
    y = pd.Series(e).groupby(pid[rows_tr]).transform("median").to_numpy()
    pf = problem_features(d, f)
    m = lgb.LGBMRegressor(**{**PARAMS, "num_leaves": int(os.environ.get("OFF_LEAVES", "63"))}).fit(
        pf.iloc[rows_tr], y, sample_weight=w
    )
    o = m.predict(pf.iloc[rows_te])
    return pd.Series(o).groupby(pid[rows_te]).transform("median").to_numpy(), m


def base_parts(x, p):
    g = M.geometry(x)
    t, pr = M.predict(g, p, parts=True)
    pr["t"] = t
    return t, g, pr


def centered_target(x, base):
    e = np.log(x.device_ns.to_numpy() / base)
    return e - pd.Series(e).groupby(x.problem_id.to_numpy()).transform("median").to_numpy()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "abl/resid"
    d = M.annotate(load(list(SETS)))
    rows = {a: np.flatnonzero((d.arch_ == a).to_numpy()) for a in ("wh", "bh")}
    base_oof, final_oof, corr_oof = (np.full(len(d), np.nan) for _ in range(3))
    for a, rr in rows.items():
        M.pin(fit, a)
        probs = d.problem_id.iloc[rr].unique()
        np.random.default_rng(0).shuffle(probs)
        for k in range(5):
            test = np.isin(d.problem_id.iloc[rr], probs[k::5])
            tr, te = rr[~test], rr[test]
            if os.environ.get("MC_TRAIN") != "1":  # MultiCore probes, when loaded, are scored but not trained on
                tr = tr[~d.mcprobe.iloc[tr].to_numpy()]
            p = fit.fit(d, tr)
            xtr, xte = d.iloc[tr], d.iloc[te]
            btr, gtr, ptr = base_parts(xtr, p)
            bte, gte, pte = base_parts(xte, p)
            ftr, fte = features(xtr, gtr, ptr), features(xte, gte, pte)
            y = centered_target(xtr, btr)
            rfrac = float(
                os.environ.get("RFRAC", "1.0")
            )  # learning curve: residual sees only this share of training problems
            if rfrac < 1.0:
                tp = xtr.problem_id.unique()
                keep_p = set(np.random.default_rng(k).choice(tp, int(len(tp) * rfrac), replace=False))
                sub = xtr.problem_id.isin(keep_p).to_numpy()
            else:
                sub = np.ones(len(xtr), bool)
            cand = xtr.origin.isin(CAND).to_numpy()
            w = 1.0 / np.sqrt(xtr.problem_id.map(xtr.problem_id.value_counts()).to_numpy())
            m = lgb.LGBMRegressor(**PARAMS).fit(ftr[cand & sub], y[cand & sub], sample_weight=w[cand & sub])
            c = SHRINK * m.predict(fte)
            for seed in range(1, int(os.environ.get("ENSEMBLE", "1"))):  # seed average: lower variance near ties
                ms = lgb.LGBMRegressor(**{**PARAMS, "random_state": seed}).fit(
                    ftr[cand & sub], y[cand & sub], sample_weight=w[cand & sub]
                )
                c = c + (SHRINK * ms.predict(fte) - c) / (seed + 1)
            off = np.zeros(len(te))
            if os.environ.get("OFFSET") == "1":
                xx = pd.concat([xtr, xte], ignore_index=True)
                ff = pd.concat([ftr, fte], ignore_index=True)
                bb = np.concatenate([btr, bte])
                cc = np.concatenate([SHRINK * m.predict(ftr), c])
                itr, ite = np.flatnonzero(cand), len(xtr) + np.arange(len(xte))
                off, _ = problem_offset(xx, ff, bb, cc, itr, ite, w[cand])
            base_oof[te], corr_oof[te], final_oof[te] = bte, c, bte * np.exp(c + off)
            print(f"{a} fold {k}: |corr| median {np.median(np.abs(c)):.3f}", flush=True)
    np.save(f"{out}_base.npy", base_oof)
    np.save(f"{out}_final.npy", final_oof)
    res = {}
    for name, pred in (("base", base_oof), ("base+resid", final_oof)):
        r = evaluate(d, pred, "", quiet=True)
        acc = rank_quality(d, pred)
        sets = {}
        for s, g in r.groupby("set", sort=False):
            sets[s] = dict(
                regret=round(gm(g.regret), 4),
                regr=int((g.vs_legacy > 1.05).sum()),
                severe=int((g.vs_legacy > 1.25).sum()),
            )
        res[name] = dict(sets=sets, pairacc=round(acc[0], 4), med_err=round(acc[1], 4))
        print(
            f"{name:11s} "
            + "  ".join(f"{s} {v['regret']:.3f}/{v['regr']}/{v['severe']}" for s, v in sets.items())
            + f" | pairacc {acc[0]:.3f} med_err {acc[1]:.3f}"
        )
    res["corr_abs_median"] = float(np.nanmedian(np.abs(corr_oof)))
    res["corr_abs_p90"] = float(np.nanquantile(np.abs(corr_oof), 0.9))
    print("correction |log| median %.3f, p90 %.3f" % (res["corr_abs_median"], res["corr_abs_p90"]))
    json.dump(res, open(f"{out}.json", "w"), indent=1)
