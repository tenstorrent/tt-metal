#!/usr/bin/env python3
"""Fit the per-layer cost model from runs.csv and write coeffs.json (format of m3_budget_sim.DEFAULT_COEFFS).

  layer_ms = a + b*W + c*max(p, p0)*h + d*h + e*n      (one segment per forward in Phase A: p = W)

p0 is an effective floor on the rows attention is charged for (few rows per chip leave cores idle);
it is grid-searched for dense; sparse attention cost does not follow rows (E2w/E3), so p0 = 0 there.

Dense is fitted from the D runs (3 layers), sparse from the S8 runs (8 layers); a stage time is
divided by its layer count, so any per-stage overhead is folded into `a`. The S0 runs are the
composition check: o = T_S8 - 8/5 * (T_S0 - T_D) per matching (W, h, n).

  analyze.py results/runs.csv results/coeffs.json
  analyze.py results_sp2/runs.csv results_sp2/coeffs_sp2.json --dense D2 --sparse S8P --exps SANITY,A1,A2,A2w,A5 \
      --overhead-sets SP2_ST1:15 --packed-exp A3 --hop-ms <Part B hop>      (SP=2 follow-up)
"""
import argparse, csv, json, sys
from collections import defaultdict

import numpy as np

SKIP_RUNS = {"e1_d_w2048"}  # first point of the process measured before warm-up settled (PROGRESS.md)
LAYERS = {"D": 3, "S8": 8, "S0": 8, "S3_7": 5, "S8_12": 5, "S11_15": 5}
# sparse-only sets with fewer / more layers than the sparse fit set: with it they give the stage overhead o(W)
OVERHEAD_SETS = {"S3_7": 5, "S8_12": 5, "S11_15": 5}
EXPS = ("SANITY", "E1", "E2", "E2c", "E2w", "LH")
DENSE, SPARSE, PACKED_EXP = "D", "S8", "E4"


def load(path):
    pts = defaultdict(list)  # (layer_set, W, h, n) -> [median ms]
    for r in csv.DictReader(open(path)):
        if r["status"] != "OK" or r["run_id"] in SKIP_RUNS or not r["wall_ms_median"]:
            continue
        if r["exp"] not in EXPS:
            continue
        cap = next((t.split("=")[1] for t in r["notes"].split() if t.startswith("capacity=")), "")
        if r["exp"] == "E2c" and cap != str(18432):
            continue  # capacity sweep: keep the smallest capacity only (cost is flat in capacity)
        seg = json.loads(r["segments_json"])[0]
        pts[(r["layer_set"], int(r["W"]), seg["h"], seg["n"])].append(float(r["wall_ms_median"]))
    return {k: float(np.median(v)) for k, v in pts.items()}


def stage_overhead(pts):
    """o(W) = T_S8 - 8 * marginal sparse layer, marginal = (T_S8 - T_Sk) / (8 - k); fit o = ob * W through 0."""
    obs = []
    L8 = LAYERS[SPARSE]
    for (ls, W, h, n), tk in pts.items():
        if ls in OVERHEAD_SETS and (SPARSE, W, h, n) in pts:
            t8, k = pts[(SPARSE, W, h, n)], OVERHEAD_SETS[ls]
            obs.append((W, h, t8 - L8 * (t8 - tk) / (L8 - k)))
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


def main(runs, out, hop_ms=0.0):
    pts = load(runs)
    ob, obs = stage_overhead(pts)
    print(f"stage overhead o(W) = {ob:.3g} ms/token * W  (from S8 vs 5-sparse-layer sets):")
    for W, h, o in obs:
        print(f"   W={W} h={h}: o={o:+.2f} ms  (fit {ob * W:.2f})")
    C = {}
    for kind, ls in (("dense", DENSE), ("sparse", SPARSE)):
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
        if ls != "S0" or (DENSE, W, h, n) not in pts or (SPARSE, W, h, n) not in pts:
            continue
        lm = lambda k, c: c["a"] + c["b"] * W + c["c"] * max(W, c["p0"]) * h + c["d"] * h + c["e"] * n
        pred = ob * W + 3 * lm("dense", C["dense"]) + 5 * lm("sparse", C["sparse"])
        os_.append((s0 - pred) / s0)
        print(f"   W={W} h={h} n={n}: S0 meas {s0:.1f}  pred {pred:.1f}  ({(s0 - pred) / s0 * 100:+.1f}%)")
    stage_check(pts, C, ob)
    # Per-segment attention overhead of a packed forward: all-cold packed (E4 C1 / C8) vs plain E1 at the same W.
    packed = {}
    for r in csv.DictReader(open(runs)):
        if r["status"] == "OK" and r["exp"] == PACKED_EXP and r["notes"].split()[0] in ("C1", "C8"):
            packed[(r["layer_set"], int(r["W"]))] = (int(r["B"]), float(r["wall_ms_median"]))
    for kind, ls in (("dense", DENSE), ("sparse", SPARSE)):
        est = []
        for (l, W), (B, ms) in packed.items():
            if l == ls and (ls, W, 0, W) in pts:
                est.append(((ms - pts[(ls, W, 0, W)]) / ((B - 1) * LAYERS[ls]), B - 1))
                print(
                    f"   {kind} W={W} B={B}: packed {ms:.1f} vs plain {pts[(ls, W, 0, W)]:.1f} -> {est[-1][0]:.3f} ms/segment/layer"
                )
        # weighted by extra segments: the B=2 point leans on a single plain run
        # clamped at 0: a packed forward cheaper than plain (SP=2) is not a per-segment saving to extrapolate
        C[kind]["seg_a"] = max(0.0, float(np.average([e for e, _ in est], weights=[w for _, w in est]))) if est else 0.0
    C["stage_overhead_ms"] = 0.0
    C["stage_overhead_per_token_ms"] = ob  # every stage in these runs embeds its own tokens
    C["hop_ms"] = hop_ms
    json.dump(C, open(out, "w"), indent=2)
    print(f"wrote {out}")


def stage_check(pts, C, ob):
    """Whole pipeline stages timed alone (layer sets '<...>_ST<k>' with the layer range in LAYER_RANGES)."""
    for (ls, W, h, n), ms in sorted(pts.items()):
        if ls not in LAYER_RANGES:
            continue
        first, cnt = LAYER_RANGES[ls]
        nd = sum(1 for L in range(first, first + cnt) if L < 3)
        lm = lambda c: c["a"] + c["b"] * W + c["c"] * max(W, c["p0"]) * h + c["d"] * h + c["e"] * n
        pred = ob * W + nd * lm(C["dense"]) + (cnt - nd) * lm(C["sparse"])
        print(f"   stage {ls} W={W} h={h}: meas {ms:.1f}  pred {pred:.1f}  ({(ms - pred) / ms * 100:+.1f}%)")


LAYER_RANGES = {}  # set from --stage-sets, e.g. SP2_ST0:0:15


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("out")
    ap.add_argument("--dense", default=DENSE)
    ap.add_argument("--sparse", default=SPARSE)
    ap.add_argument("--exps", default=",".join(EXPS))
    ap.add_argument("--overhead-sets", default=None, help="NAME:layers,... sparse-only sets for o(W)")
    ap.add_argument("--packed-exp", default=PACKED_EXP)
    ap.add_argument("--stage-sets", default="", help="NAME:first:count,... whole stages to check (not fitted)")
    ap.add_argument("--hop-ms", type=float, default=0.0)
    a = ap.parse_args()
    DENSE, SPARSE, PACKED_EXP, EXPS = a.dense, a.sparse, a.packed_exp, tuple(a.exps.split(","))
    LAYERS.setdefault(DENSE, 3)
    LAYERS.setdefault(SPARSE, 8)
    if a.overhead_sets:
        OVERHEAD_SETS = {k: int(v) for k, v in (x.split(":") for x in a.overhead_sets.split(","))}
        LAYERS.update(OVERHEAD_SETS)
    for x in filter(None, a.stage_sets.split(",")):
        name, first, cnt = x.split(":")
        LAYER_RANGES[name] = (int(first), int(cnt))
    main(a.runs, a.out, a.hop_ms)
