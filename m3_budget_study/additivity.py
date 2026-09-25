#!/usr/bin/env python3
"""E4 additivity and coupling from runs.csv.

  T_pred(C) = T(W, all cold) + sum_i [T1(h_i, n_i) - T1(0, 2048)]

T1 = E2 single-segment W=2048 time for the same layer set, linear in h between grid points at the same n.
T(W, all cold) = the packed all-cold composition of the same run (C1 at W=4096, C8 at W=8192).

  additivity.py results/runs.csv [results/additivity.csv]
"""
import csv, json, sys
from collections import defaultdict


def load(path):
    t1 = defaultdict(dict)  # layer_set -> {(h, n): ms}
    packed = {}  # (layer_set, W, compo) -> (segments, ms)
    for r in csv.DictReader(open(path)):
        if r["status"] != "OK" or not r["wall_ms_median"]:
            continue
        segs = json.loads(r["segments_json"])
        if r["exp"] == "E2" and int(r["W"]) == 2048:
            t1[r["layer_set"]][(segs[0]["h"], segs[0]["n"])] = float(r["wall_ms_median"])
        elif r["exp"] == "E4":
            packed[(r["layer_set"], int(r["W"]), r["notes"].split()[0])] = (segs, float(r["wall_ms_median"]))
    return t1, packed


def T1(grid, h, n):
    hs = sorted(hh for hh, nn in grid if nn == n)
    if h in hs:
        return grid[(h, n)]
    lo = max(x for x in hs if x <= h)
    hi = min(x for x in hs if x >= h)
    f = (h - lo) / (hi - lo)
    return grid[(lo, n)] + f * (grid[(hi, n)] - grid[(lo, n)])


def main(path, out=None):
    t1, packed = load(path)
    rows = []
    for (ls, W, name), (segs, meas) in sorted(packed.items()):
        cold = packed.get((ls, W, "C1" if W == 4096 else "C8"))
        if cold is None or ls not in t1:
            continue
        pred = cold[1] + sum(T1(t1[ls], s["h"], s["n"]) - T1(t1[ls], 0, 2048) for s in segs)
        rows.append(
            dict(
                layer_set=ls,
                W=W,
                compo=name,
                segments=" ".join(f"{s['h']}:{s['n']}" for s in segs),
                meas_ms=round(meas, 2),
                pred_ms=round(pred, 2),
                resid_pct=round((meas - pred) / meas * 100, 1),
            )
        )
    print(f"{'set':4} {'W':>5} {'compo':6} {'meas':>8} {'pred':>8} {'resid%':>7}  segments")
    for r in rows:
        print(
            f"{r['layer_set']:4} {r['W']:5d} {r['compo']:6} {r['meas_ms']:8.1f} {r['pred_ms']:8.1f} {r['resid_pct']:+7.1f}  {r['segments']}"
        )
    if rows:
        print(f"max |residual| {max(abs(r['resid_pct']) for r in rows):.1f}%")
    print("\ncoupling (W=4096):")
    for ls in sorted({k[0] for k in packed}):
        g = lambda c: packed.get((ls, 4096, c), (None, None))[1]
        if None not in (g("C1"), g("C2"), g("C5")):
            print(
                f"  {ls}: C2 + C1 = {g('C2') + g('C1'):.1f} ms  vs  2 x C5 = {2 * g('C5'):.1f} ms"
                f"   (per-forward {g('C2'):.1f}/{g('C1'):.1f} vs {g('C5'):.1f}/{g('C5'):.1f})"
            )
        if None not in (g("C4"), g("C4r")):
            print(
                f"  {ls}: loop order C4 {g('C4'):.1f} vs C4r {g('C4r'):.1f} ms ({(g('C4r') - g('C4')) / g('C4') * 100:+.1f}%)"
            )
        if None not in (g("C1"), g("E5p")):
            print(
                f"  {ls}: padding 0:2048,0:256 {g('E5p'):.1f} vs 0:2048,0:2048 {g('C1'):.1f} ms"
                f" ({(g('E5p') - g('C1')) / g('C1') * 100:+.1f}%)"
            )
    if out and rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:3])
