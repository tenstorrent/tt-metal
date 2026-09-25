#!/usr/bin/env python3
"""Fit the per-layer cost model from runs.csv and write coeffs.json (format of m3_budget_sim.DEFAULT_COEFFS).

  layer_ms = a + b*W + c*max(p, p0)*h + d*h + e*n      (one segment per forward in Phase A: p = W)

p0 is an effective floor on the rows attention is charged for (few rows per chip leave cores idle);
it is grid-searched for dense; sparse attention cost does not follow rows (E2w/E3), so p0 = 0 there.

Dense is fitted from the D runs (3 layers), sparse from the S8 runs (8 layers); a stage time is
divided by its layer count, so any per-stage overhead is folded into `a`. The S0 runs are the
composition check: o = T_S8 - 8/5 * (T_S0 - T_D) per matching (W, h, n).

  analyze.py results/runs.csv results/coeffs.json
"""
import csv, json, sys
from collections import defaultdict

import numpy as np

SKIP_RUNS = {"e1_d_w2048"}  # first point of the process measured before warm-up settled (PROGRESS.md)
LAYERS = {"D": 3, "S8": 8, "S0": 8, "S3_7": 5, "S8_12": 5, "S11_15": 5}
FIVE = ("S3_7", "S8_12", "S11_15")  # 5 sparse layers: with S8 they give the stage overhead o(W)


def load(path):
    pts = defaultdict(list)  # (layer_set, W, h, n) -> [median ms]
    for r in csv.DictReader(open(path)):
        if r["status"] != "OK" or r["run_id"] in SKIP_RUNS or not r["wall_ms_median"]:
            continue
        if r["exp"] not in ("SANITY", "E1", "E2", "E2c", "E2w", "LH"):
            continue
        cap = next((t.split("=")[1] for t in r["notes"].split() if t.startswith("capacity=")), "")
        if r["exp"] == "E2c" and cap != str(18432):
            continue  # capacity sweep: keep the smallest capacity only (cost is flat in capacity)
        seg = json.loads(r["segments_json"])[0]
        pts[(r["layer_set"], int(r["W"]), seg["h"], seg["n"])].append(float(r["wall_ms_median"]))
    return {k: float(np.median(v)) for k, v in pts.items()}


def stage_overhead(pts):
    """o(W) = T_S8 - 8 * marginal sparse layer, marginal = (T_S8 - T_S5) / 3; fit o = ob * W through 0."""
    obs = []
    for (ls, W, h, n), t5 in pts.items():
        if ls in FIVE and ("S8", W, h, n) in pts:
            t8 = pts[("S8", W, h, n)]
            obs.append((W, h, t8 - 8 * (t8 - t5) / 3))
    ob = sum(o * W for W, _, o in obs) / sum(W * W for W, _, _ in obs)
    return ob, sorted(obs)


def fit(pts, layer_set, ob=0.0, p0_grid=(0,)):
    rows = [(W, h, n, (ms - ob * W) / LAYERS[layer_set]) for (ls, W, h, n), ms in pts.items() if ls == layer_set]
    y = np.array([v for *_, v in rows])
    best = None
    for p0 in p0_grid:
        X = np.array([[1.0, W, max(W, p0) * h, h, n] for W, h, n, _ in rows])
        keep = list(range(5))
        while True:  # non-negative b..e: drop a feature whose coefficient goes negative, refit
            sub, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
            neg = [k for k, v in zip(keep, sub) if k > 0 and v < 0]
            if not neg:
                break
            keep.remove(min(neg, key=lambda k: sub[keep.index(k)]))
        coef = np.zeros(5)
        coef[keep] = sub
        ss_res = float(((y - X @ coef) ** 2).sum())
        if best is None or ss_res < best[0]:
            best = (ss_res, p0, coef, X @ coef)
    ss_res, p0, coef, pred = best
    ss_tot = float(((y - y.mean()) ** 2).sum())
    resid = sorted(
        (
            (abs(p - v) / v, W, h, n, v * LAYERS[layer_set] + ob * W, p * LAYERS[layer_set] + ob * W)
            for (W, h, n, v), p in zip(rows, pred)
        ),
        reverse=True,
    )
    return dict(zip("abcde", map(float, coef)), p0=p0), 1 - ss_res / ss_tot, resid


def main(runs, out):
    pts = load(runs)
    ob, obs = stage_overhead(pts)
    print(f"stage overhead o(W) = {ob:.3g} ms/token * W  (from S8 vs 5-sparse-layer sets):")
    for W, h, o in obs:
        print(f"   W={W} h={h}: o={o:+.2f} ms  (fit {ob * W:.2f})")
    C = {}
    for kind, ls in (("dense", "D"), ("sparse", "S8")):
        coef, r2, resid = fit(pts, ls, ob, range(0, 8193, 128) if kind == "dense" else (0,))
        C[kind] = coef
        print(
            f"{kind} ({ls}, {len(resid)} points): "
            + "  ".join(f"{k}={v:.4g}" for k, v in coef.items())
            + f"  R2={r2:.4f}"
        )
        for rel, W, h, n, meas, pred in resid[:5]:
            print(f"   worst: W={W} h={h} n={n}  meas {meas:.1f}  pred {pred:.1f}  ({rel*100:+.1f}%)")
    print("S0 composition check: measured vs o(W) + 3 dense + 5 sparse from the fit")
    os_ = []
    for (ls, W, h, n), s0 in sorted(pts.items()):
        if ls != "S0" or ("D", W, h, n) not in pts or ("S8", W, h, n) not in pts:
            continue
        lm = lambda k, c: c["a"] + c["b"] * W + c["c"] * max(W, c["p0"]) * h + c["d"] * h + c["e"] * n
        pred = ob * W + 3 * lm("dense", C["dense"]) + 5 * lm("sparse", C["sparse"])
        os_.append((s0 - pred) / s0)
        print(f"   W={W} h={h} n={n}: S0 meas {s0:.1f}  pred {pred:.1f}  ({(s0 - pred) / s0 * 100:+.1f}%)")
    # Per-segment attention overhead of a packed forward: all-cold packed (E4 C1 / C8) vs plain E1 at the same W.
    packed = {}
    for r in csv.DictReader(open(runs)):
        if r["status"] == "OK" and r["exp"] == "E4" and r["notes"].split()[0] in ("C1", "C8"):
            packed[(r["layer_set"], int(r["W"]))] = (int(r["B"]), float(r["wall_ms_median"]))
    for kind, ls in (("dense", "D"), ("sparse", "S8")):
        est = []
        for (l, W), (B, ms) in packed.items():
            if l == ls and (ls, W, 0, W) in pts:
                est.append(((ms - pts[(ls, W, 0, W)]) / ((B - 1) * LAYERS[ls]), B - 1))
                print(
                    f"   {kind} W={W} B={B}: packed {ms:.1f} vs plain {pts[(ls, W, 0, W)]:.1f} -> {est[-1][0]:.3f} ms/segment/layer"
                )
        # weighted by extra segments: the B=2 point leans on a single plain run
        C[kind]["seg_a"] = float(np.average([e for e, _ in est], weights=[w for _, w in est])) if est else 0.0
    C["stage_overhead_ms"] = 0.0
    C["stage_overhead_per_token_ms"] = ob  # every stage in these runs embeds its own tokens
    C["hop_ms"] = 0.0
    json.dump(C, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
