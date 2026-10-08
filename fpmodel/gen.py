"""How well does the model extend beyond its training data? Offline checks (no device).

  random    5-fold by problem (the baseline in the results so far); also keeps each fold's constants
  grouped   leave one regime out: fit without every problem in a group, predict that group
            (memory layout, B dtype, fidelity, size decade, batched, epilogue, M size)
  family    fit without one family's rows (5-fold by problem), score all picks
  cross     fit on WH designed only -> WH targeted; BH constants -> WH rows (WH spec), WH constants -> BH rows
  curve     fit on 10 / 25 / 50% of the training problems of each fold
Picks use the 5% shallow-K tie-break throughout.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, pickle

sys.path.insert(0, ".")
from fit import *

MARGIN = 0.05


def pick_score(d, pred):
    s = np.empty(len(d))
    x = d.assign(pred=pred)
    for _, g in x.groupby("problem_id", sort=False):
        c = g.origin.isin(CAND).to_numpy()
        best = g.pred[c].min()
        near = (g.pred <= best * (1 + MARGIN)).to_numpy() & c
        s[g.index] = np.where(near, g.in0_block_w.fillna(1).to_numpy() * 1e6 + g.pred / best, 1e12 + g.pred)
    return s


def score(d, pred, probs=None):
    """per-problem results for the problems in probs (all if None)"""
    r = evaluate(d, pick_score(d, pred), "", quiet=True)
    return r if probs is None else r[r.problem_id.isin(probs)]


def summ(r):
    return dict(
        n=len(r),
        regret=gm(r.regret),
        regr=int((r.vs_legacy > 1.05).sum()),
        rules_regret=gm(r.rules_regret.dropna()),
        rules_regr=int((r.rules_vs_legacy > 1.05).sum()),
        worst=float(r.vs_legacy.max()),
    )


def fmt(s):
    return (
        f"n={s['n']:4d}  regret {s['regret']:.3f}  regr {s['regr']:3d}  worst/legacy {s['worst']:.2f}"
        f"   | rules {s['rules_regret']:.3f} / {s['rules_regr']}"
    )


def groups(d):
    """problem-level regime labels"""
    p = d.groupby("problem_id").first()
    best_us = p.best / 1e3
    mt = np.ceil(p.M / 32)
    return {
        "a_mem": p.a_mem.where(p.a_mem.isin(["dram", "l1"]), "sharded"),
        "b_dtype": p.b_dtype,
        "fidelity": p.fidelity,
        "size": pd.cut(best_us, [0, 30, 300, 3000, np.inf], labels=["<30us", "30-300us", "0.3-3ms", ">3ms"]).astype(
            str
        ),
        "batched": np.where(p.batch > 1, "batch>1", "batch=1"),
        "epilogue": np.where((p.bias.fillna(0) == 1) | p.activation.notna(), "bias/act", "none"),
        "M": pd.cut(mt, [0, 4, 32, np.inf], labels=["Mt<=4", "Mt 5-32", "Mt>32"]).astype(str),
    }


if __name__ == "__main__":
    d = load(list(SETS))
    arch = d.arch_.to_numpy()
    rows = {a: np.flatnonzero(arch == a) for a in ("wh", "bh")}
    folds = {}
    for a, rr in rows.items():
        probs = d.problem_id.iloc[rr].unique()
        rng = np.random.default_rng(0)
        rng.shuffle(probs)
        folds[a] = [probs[k::5] for k in range(5)]
    out = {}

    # random 5-fold baseline, keeping each fold's constants
    base = np.full(len(d), np.nan)
    fold_p = {a: [] for a in rows}
    for a, rr in rows.items():
        for f in folds[a]:
            test = np.isin(d.problem_id.iloc[rr], f)
            p = fit(d, rr[~test])
            fold_p[a].append(p)
            base[rr[test]] = predict_rows(d, rr[test], p)
    rb = score(d, base)
    out["random"] = {s: summ(g) for s, g in rb.groupby("set")}
    print("== random 5-fold (baseline)")
    for s, v in out["random"].items():
        print(f"  {s:12s} {fmt(v)}")

    # constant stability across folds
    print("\n== constants across the 5 folds: median, spread (max/min)")
    out["stability"] = {}
    for a in rows:
        P = pd.DataFrame(fold_p[a])
        out["stability"][a] = P
        print(f"  {a}: " + "  ".join(f"{k} {P[k].median():.3g} x{P[k].max() / max(P[k].min(), 1e-12):.2g}" for k in P))

    # leave one regime out
    G = groups(d)
    print("\n== leave one regime out (same problems: random-fold vs regime held out)")
    out["grouped"] = {}
    for name, lab in G.items():
        lab = pd.Series(np.asarray(lab), index=G["b_dtype"].index)
        print(f"  -- {name}")
        for a, rr in rows.items():
            ap = d.problem_id.iloc[rr]
            for v in sorted(lab[lab.index.isin(ap)].unique()):
                held = lab.index[(lab == v).to_numpy() & lab.index.isin(ap)]
                if len(held) < 5 or len(held) > 0.8 * ap.nunique():
                    continue
                test = ap.isin(held).to_numpy()
                p = fit(d, rr[~test])
                pr = base.copy()
                pr[rr[test]] = predict_rows(d, rr[test], p)
                sg, sr = summ(score(d, pr, held)), summ(rb[rb.problem_id.isin(held)])
                out["grouped"][(name, a, v)] = dict(held=sg, random=sr)
                print(
                    f"     {a} {v:10s} n={sg['n']:4d}  regret {sr['regret']:.3f} -> {sg['regret']:.3f}   regr {sr['regr']:3d} -> {sg['regr']:3d}"
                    f"   (rules {sr['rules_regret']:.3f} / {sr['rules_regr']})"
                )

    # leave one family out of training
    print("\n== family left out of training (5-fold by problem), all picks scored")
    out["family"] = {}
    fam = d.family.to_numpy()
    for f in ("2d", "1d_in0", "1d_in1", "reuse"):
        pr = np.full(len(d), np.nan)
        for a, rr in rows.items():
            for fp in folds[a]:
                test = np.isin(d.problem_id.iloc[rr], fp)
                tr = rr[~test & (fam[rr] != f)]
                pr[rr[test]] = predict_rows(d, rr[test], fit(d, tr))
        r = score(d, pr)
        out["family"][f] = {s: summ(g) for s, g in r.groupby("set")}
        print(
            f"  without {f:7s} " + "  ".join(f"{s} {v['regret']:.3f}/{v['regr']}" for s, v in out["family"][f].items())
        )

    # cross-set / cross-arch
    print("\n== cross-set and cross-arch")
    whd = np.flatnonzero((d.set == "wh_designed").to_numpy())
    wht = np.flatnonzero((d.set == "wh_target").to_numpy())
    p = fit(d, whd)
    pr = base.copy()
    pr[wht] = predict_rows(d, wht, p)
    tp = d.problem_id.iloc[wht].unique()
    out["designed->target"] = summ(score(d, pr, tp))
    print(
        f"  WH designed -> WH target   {fmt(out['designed->target'])}   (random-fold {out['random']['wh_target']['regret']:.3f} / {out['random']['wh_target']['regr']})"
    )
    full = {a: fit(d, rr) for a, rr in rows.items()}
    for src, dst in (("wh", "bh"), ("bh", "wh")):
        pr = base.copy()
        pr[rows[dst]] = predict_rows(d, rows[dst], full[src])
        r = score(d, pr, d.problem_id.iloc[rows[dst]].unique())
        out[f"{src}->{dst}"] = {s: summ(g) for s, g in r.groupby("set")}
        for s, v in out[f"{src}->{dst}"].items():
            print(f"  {src} constants -> {dst} spec  {s:12s} {fmt(v)}")

    # learning curve
    print("\n== learning curve: fraction of each fold's training problems")
    out["curve"] = {}
    for frac in (0.1, 0.25, 0.5):
        pr = np.full(len(d), np.nan)
        ps = {a: [] for a in rows}
        for a, rr in rows.items():
            for k, fp in enumerate(folds[a]):
                trp = np.concatenate([folds[a][j] for j in range(5) if j != k])
                rng = np.random.default_rng(k)
                sub = rng.choice(trp, max(5, int(frac * len(trp))), replace=False)
                tr = rr[np.isin(d.problem_id.iloc[rr], sub)]
                test = rr[np.isin(d.problem_id.iloc[rr], fp)]
                p = fit(d, tr)
                ps[a].append(p)
                pr[test] = predict_rows(d, test, p)
        r = score(d, pr)
        out["curve"][frac] = {s: summ(g) for s, g in r.groupby("set")}
        print(
            f"  {frac:4.0%} " + "  ".join(f"{s} {v['regret']:.3f}/{v['regr']}" for s, v in out["curve"][frac].items())
        )
    pickle.dump(out, open("gen_results.pkl", "wb"))
