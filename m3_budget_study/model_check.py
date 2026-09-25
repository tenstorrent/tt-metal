#!/usr/bin/env python3
"""Predict every packed E4 composition from coeffs.json (the simulator's cost model) and report residuals.

  model_check.py results/runs.csv results/coeffs.json
"""
import csv, json, sys

N = {"D": (3, 0), "S8": (0, 8), "S0": (3, 5)}


def stage_ms(C, n_dense, n_sparse, W, segs):
    t = C.get("stage_overhead_per_token_ms", 0.0) * W
    for kind, cnt in (("dense", n_dense), ("sparse", n_sparse)):
        k = C[kind]
        per = k["a"] - k.get("seg_a", 0.0) + k["b"] * W
        for s in segs:
            p = -(-s["n"] // 2048) * 2048
            per += k.get("seg_a", 0.0) + k["c"] * max(p, k.get("p0", 0)) * s["h"] + k["d"] * s["h"] + k["e"] * s["n"]
        t += cnt * per
    return t


def main(runs, coeffs):
    C = json.load(open(coeffs))
    res = []
    for r in csv.DictReader(open(runs)):
        if r["status"] != "OK" or r["exp"] != "E4":
            continue
        segs = json.loads(r["segments_json"])
        pred = stage_ms(C, *N[r["layer_set"]], int(r["W"]), segs)
        meas = float(r["wall_ms_median"])
        res.append((r["layer_set"], int(r["W"]), r["notes"].split()[0], meas, pred, (meas - pred) / meas * 100))
    for ls, W, name, m, p, e in sorted(res):
        print(f"{ls:3} {W:5d} {name:4} meas {m:7.1f} model {p:7.1f} ({e:+5.1f}%)")
    print(
        f"max |resid| {max(abs(x[-1]) for x in res):.1f}%  mean |resid| {sum(abs(x[-1]) for x in res) / len(res):.1f}%"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
