#!/usr/bin/env python3
"""Fit the per-layer cost model from runs.csv and write coeffs.json (format of m3_budget_sim.DEFAULT_COEFFS).

  layer_ms = a + b*W + c*p*h + d*h + e*n      (one segment per forward in Phase A: p = W)

Dense is fitted from the D runs (3 layers), sparse from the S8 runs (8 layers); a stage time is
divided by its layer count, so any per-stage overhead is folded into `a`. The S0 runs are the
composition check: o = T_S8 - 8/5 * (T_S0 - T_D) per matching (W, h, n).

  analyze.py results/runs.csv results/coeffs.json
"""
import csv, json, sys
from collections import defaultdict

import numpy as np

SKIP_RUNS = {"e1_d_w2048"}  # first point of the process measured before warm-up settled (PROGRESS.md)
LAYERS = {"D": 3, "S8": 8, "S0": 8}


def load(path):
    pts = defaultdict(list)  # (layer_set, W, h, n) -> [median ms]
    for r in csv.DictReader(open(path)):
        if r["status"] != "OK" or r["run_id"] in SKIP_RUNS or not r["wall_ms_median"]:
            continue
        if r["exp"] not in ("SANITY", "E1", "E2", "E2c", "E2w"):
            continue
        cap = next((t.split("=")[1] for t in r["notes"].split() if t.startswith("capacity=")), "")
        if r["exp"] == "E2c" and cap != str(18432):
            continue  # capacity sweep: keep the smallest capacity only (cost is flat in capacity)
        seg = json.loads(r["segments_json"])[0]
        pts[(r["layer_set"], int(r["W"]), seg["h"], seg["n"])].append(float(r["wall_ms_median"]))
    return {k: float(np.median(v)) for k, v in pts.items()}


def fit(pts, layer_set):
    rows = [(W, h, n, ms / LAYERS[layer_set]) for (ls, W, h, n), ms in pts.items() if ls == layer_set]
    X = np.array([[1.0, W, W * h, h, n] for W, h, n, _ in rows])
    y = np.array([v for *_, v in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    resid = sorted(
        (
            (abs(p - v) / v, W, h, n, v * LAYERS[layer_set], p * LAYERS[layer_set])
            for (W, h, n, v), p in zip(rows, pred)
        ),
        reverse=True,
    )
    return dict(zip("abcde", map(float, coef))), 1 - ss_res / ss_tot, resid


def main(runs, out):
    pts = load(runs)
    C = {}
    for kind, ls in (("dense", "D"), ("sparse", "S8")):
        coef, r2, resid = fit(pts, ls)
        C[kind] = coef
        print(
            f"{kind} ({ls}, {len(resid)} points): "
            + "  ".join(f"{k}={v:.4g}" for k, v in coef.items())
            + f"  R2={r2:.4f}"
        )
        for rel, W, h, n, meas, pred in resid[:5]:
            print(f"   worst: W={W} h={h} n={n}  meas {meas:.1f}  pred {pred:.1f}  ({rel*100:+.1f}%)")
    print("stage overhead o = T_S8 - 8/5 (T_S0 - T_D):")
    os_ = []
    for (ls, W, h, n), s0 in sorted(pts.items()):
        if ls != "S0" or ("D", W, h, n) not in pts or ("S8", W, h, n) not in pts:
            continue
        o = pts[("S8", W, h, n)] - 8 / 5 * (s0 - pts[("D", W, h, n)])
        os_.append(o)
        print(
            f"   W={W} h={h} n={n}: S0={s0:.1f} D={pts[('D', W, h, n)]:.1f} S8={pts[('S8', W, h, n)]:.1f}  o={o:+.1f}"
        )
    C["stage_overhead_ms"] = 0.0  # folded into `a`; o above is reported, not added
    C["hop_ms"] = 0.0
    json.dump(C, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
