#!/usr/bin/env python3
"""Offline analysis for the picker-generalization campaign.

Loads the sweep JSONL results and compares three pickers per shape:
  - MEASURED OPTIMUM : the best ok config actually measured (lower bound / oracle),
  - PRODUCTION        : the current C++ auto_select_config choice (measured via config=None baseline),
  - PROPOSED          : the transparent hierarchical heuristic in regime_a_model.propose_config().

Reports, split by train/val/holdout and the FLUX/LTX subset: geometric-mean gap, median gap, worst gap,
count of >3% and >5% misses, and per-shape proposed configs. Gap = wall(picker)/wall(measured-optimum) - 1.

The PROPOSED heuristic is refined against the TRAIN split only; val/holdout/FLUX-LTX are reported as
generalization checks. This module makes NO C++ change — it is the offline mirror only.

Usage:
  python3 analyze.py                 # full report from results/*.jsonl
  python3 analyze.py --md report.md  # also write the Markdown report
"""
import argparse, json, os, glob, statistics, math
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{HERE}/results"
sys.path.insert(0, HERE)
import regime_a_model as model  # noqa: E402
import corpus as corpus_mod  # noqa: E402

TILE = 32


def split_of(M, K, N):
    for sp, lst in corpus_mod.SWEPT.items():
        if (M, K, N) in lst:
            return sp
    return "?"


def is_fluxltx(Mt, Kt, Nt):
    return (Mt, Kt, Nt) in model.KTABLE


def _median_wall(records):
    """Median over all timed samples across relaunches for one config's records (ok only)."""
    samples = [s for r in records if r["outcome"] == "ok" for s in (r["samples"] or ([r["wall_us"]] if r["wall_us"] else []))]
    return statistics.median(samples) if samples else None


def load_shape(M, K, N):
    """Return dict: cfgkey -> {'Pk':..,'wall':median_us,'pcc':best_pcc,'outcome_hist':{}} merged over
    initial+rerun records. Rerun (with PCC) supersedes for wall when present."""
    p = f"{RESULTS}/sweep_{M}x{K}x{N}.jsonl"
    if not os.path.exists(p):
        return {}
    by = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by.setdefault(r["cfgkey"], []).append(r)
    out = {}
    for k, recs in by.items():
        rerun = [r for r in recs if r.get("tag") == "rerun"]
        initial = [r for r in recs if r.get("tag") == "initial"]
        wall = _median_wall(rerun) or _median_wall(initial)
        pccs = [r["pcc"] for r in recs if r.get("pcc") is not None]
        hist = {}
        for r in recs:
            hist[r["outcome"]] = hist.get(r["outcome"], 0) + 1
        ex = recs[0]
        out[k] = {"cfgkey": k, "Pk": ex["Pk"], "Ns": ex["Ns"], "Sm": ex["Sm"], "kb": ex["kb"],
                  "nsb": ex["nsb"], "wall": wall, "pcc": (min(pccs) if pccs else None),
                  "outcome": hist, "has_rerun": bool(rerun)}
    return out


def shape_summary(M, K, N):
    """Per-shape: measured optimum, production wall (config=None baseline), and the config dicts."""
    data = load_shape(M, K, N)
    oks = {k: v for k, v in data.items() if v["wall"] is not None and k != "None"}
    if not oks:
        return None
    best_k = min(oks, key=lambda k: oks[k]["wall"])
    best = oks[best_k]
    prod = data.get("None")
    return {"M": M, "K": K, "N": N, "data": data, "oks": oks,
            "best_key": best_k, "best_wall": best["wall"], "best_cfg": best,
            "prod_wall": prod["wall"] if prod else None}


def wall_of_config(oks, cfg):
    """Look up the measured wall of a proposed (Pk,Ns,Sm,kb,nsb); None if that exact config wasn't swept
    (e.g. nsb off the lattice) — the caller falls back to the nearest lattice nsb."""
    k = f"{cfg[0]},{cfg[1]},{cfg[2]},{cfg[3]},{cfg[4]}"
    if k in oks:
        return oks[k]["wall"], k, False
    # nearest-nsb fallback within same (Pk,Ns,Sm,kb): the lattice may not contain the proposed nsb.
    cands = [(v, kk) for kk, v in oks.items()
             if (v["Pk"], v["Ns"], v["Sm"], v["kb"]) == (cfg[0], cfg[1], cfg[2], cfg[3])]
    if cands:
        v, kk = min(cands, key=lambda z: abs(z[0]["nsb"] - cfg[4]))
        return v["wall"], kk, True
    return None, None, True


def gaps_report(summaries, picker_fn, label):
    """picker_fn(Mt,Kt,Nt)->cfg tuple. Returns per-shape gaps + aggregate metrics.

    For 'production' the wall is the DIRECTLY-MEASURED config=None baseline (prod_wall) — the true
    production result — not a config lookup. For other pickers the wall is looked up in the swept set
    (nearest-nsb fallback if the exact nsb was off the lattice)."""
    rows = []
    for s in summaries:
        Mt, Kt, Nt = s["M"] // TILE, s["K"] // TILE, s["N"] // TILE
        cfg = picker_fn(Mt, Kt, Nt)
        if label == "production" and s["prod_wall"] is not None:
            wall, matched_k, approx = s["prod_wall"], "None(baseline)", False
        else:
            wall, matched_k, approx = wall_of_config(s["oks"], cfg)
        gap = (wall / s["best_wall"] - 1.0) if (wall and s["best_wall"]) else None
        rows.append({"shape": f"{s['M']}x{s['K']}x{s['N']}", "split": split_of(s["M"], s["K"], s["N"]),
                     "fluxltx": is_fluxltx(Mt, Kt, Nt), "cfg": list(cfg), "matched": matched_k,
                     "approx": approx, "wall": wall, "best": s["best_wall"], "gap": gap})
    return rows


def agg(rows):
    g = [r["gap"] for r in rows if r["gap"] is not None]
    if not g:
        return {"n": 0}
    geo = math.exp(statistics.mean(math.log(1 + x) for x in g)) - 1
    return {"n": len(g), "geomean_gap": geo, "median_gap": statistics.median(g),
            "worst_gap": max(g), "n_gt3": sum(1 for x in g if x > 0.03),
            "n_gt5": sum(1 for x in g if x > 0.05)}


def production_fn(Mt, Kt, Nt):
    return model.production_pick(Mt, Kt, Nt)[:5]


def proposed_fn(Mt, Kt, Nt):
    return model.propose_config(Mt, Kt, Nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    all_shapes = [(M, K, N) for lst in corpus_mod.SWEPT.values() for (M, K, N) in lst]
    summaries = [s for s in (shape_summary(M, K, N) for (M, K, N) in all_shapes) if s]
    print(f"loaded {len(summaries)}/{len(all_shapes)} shapes with >=1 ok config\n")

    have_proposed = hasattr(model, "propose_config")
    pickers = [("production", production_fn)] + ([("proposed", proposed_fn)] if have_proposed else [])

    lines = []
    for label, fn in pickers:
        rows = gaps_report(summaries, fn, label)
        print(f"===== {label} vs measured optimum =====")
        for subset, pred in [("ALL", lambda r: True), ("train", lambda r: r["split"] == "train"),
                             ("val", lambda r: r["split"] == "val"), ("holdout", lambda r: r["split"] == "holdout"),
                             ("fluxltx", lambda r: r["fluxltx"])]:
            a = agg([r for r in rows if pred(r)])
            if a.get("n"):
                print(f"  {subset:8s} n={a['n']:2d}  geomean {a['geomean_gap']*100:5.1f}%  "
                      f"median {a['median_gap']*100:5.1f}%  worst {a['worst_gap']*100:5.1f}%  "
                      f">3%={a['n_gt3']} >5%={a['n_gt5']}")
                lines.append(f"| {label} | {subset} | {a['n']} | {a['geomean_gap']*100:.1f}% | "
                             f"{a['median_gap']*100:.1f}% | {a['worst_gap']*100:.1f}% | {a['n_gt3']} | {a['n_gt5']} |")
        # per-shape detail (worst offenders)
        worst = sorted((r for r in rows if r["gap"] is not None), key=lambda r: -r["gap"])[:8]
        print("  worst shapes:")
        for r in worst:
            print(f"    {r['shape']:16s} [{r['split']:7s}] gap {r['gap']*100:5.1f}%  cfg {r['cfg']}  "
                  f"{'(nsb approx '+r['matched']+')' if r['approx'] else ''}")
        print()

    if args.md:
        with open(args.md, "w") as f:
            f.write("# Picker comparison (gap vs measured optimum)\n\n")
            f.write("| picker | subset | n | geomean | median | worst | >3% | >5% |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            f.write("\n".join(lines) + "\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
