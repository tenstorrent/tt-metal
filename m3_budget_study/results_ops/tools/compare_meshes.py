#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side per-op table of two meshes' zone profiles (per_op.csv rows from zones_to_per_op.py).

  compare_meshes.py --a per_op_4x2.csv:p4x2_w4096_h141312_prose --b per_op.csv:p0a_w4096_h141312_prose \\
      [--layers 3,4,5,6] [--zones-a profiles/<run>/zones.json] [--chips-a 8 --chips-b 8] [--tokens 4096]

Per op: mean over the selected layers of the worst-chip ms, eff = roof / worst, share = worst / layer worst.
With --zones-a, the harness-only sub-zones under attn/ag_kv (head_slice / head_concat) are listed separately and
an "ag_kv (native est.)" row = ag_kv - those copies is added. chip-us/token = layer ms x chips / tokens.
"""

import argparse
import csv
import json
from collections import defaultdict

SUB = ("head_slice", "head_concat")


def load(spec, layers):
    path, run = spec.rsplit(":", 1)
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["run_id"] == run and int(r["layer"]) in layers:
            for k in ("worst_ms", "mean_ms", "roof_ms", "share_of_layer_worst"):
                if r[k] != "":
                    acc[r["op"]][k].append(float(r[k]))
    return {op: {k: sum(v) / len(v) for k, v in d.items()} for op, d in acc.items()}


def sub_zones(path, layers):
    """{sub: mean over layers of the per-layer worst-chip ms} for the ag_kv sub-zones."""
    z = json.load(open(path))["zones"]
    out = {}
    for s in SUB:
        v = [z[k]["ms_max"] for k in z if k.endswith(f"/attn/ag_kv/{s}") and int(k.split("/")[1][5:7]) in layers]
        if v:
            out[s] = sum(v) / len(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="csv:run_id")
    ap.add_argument("--b", required=True, help="csv:run_id")
    ap.add_argument("--layers", default="3,4,5,6")
    ap.add_argument("--zones-a")
    ap.add_argument("--chips-a", type=int, default=8)
    ap.add_argument("--chips-b", type=int, default=8)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--labels", default="A,B")
    args = ap.parse_args()
    layers = {int(x) for x in args.layers.split(",")}
    A, B = load(args.a, layers), load(args.b, layers)
    la, lb = args.labels.split(",")
    if args.zones_a:
        sz = sub_zones(args.zones_a, layers)
        for s, v in sz.items():
            A[f"  ag_kv/{s}"] = {"worst_ms": v}
        if sz and "ag_kv" in A:
            A["ag_kv (native est.)"] = {
                "worst_ms": A["ag_kv"]["worst_ms"] - sum(sz.values()),
                "roof_ms": A["ag_kv"].get("roof_ms"),
            }
    order = [op for op in A if op != "layer_total"] + [op for op in B if op not in A and op != "layer_total"]
    order += ["layer_total"]
    lt_a, lt_b = A.get("layer_total", {}).get("worst_ms", 0), B.get("layer_total", {}).get("worst_ms", 0)

    def cells(d, lt):
        if not d:
            return ["", "", ""]
        w, r = d.get("worst_ms"), d.get("roof_ms")
        eff = f"{r / w:.0%}" if r and w else ""
        return [f"{w:.3f}", eff, f"{w / lt:.1%}" if lt else ""]

    print(f"| op | {la} ms | {la} eff | {la} share | {lb} ms | {lb} eff | {lb} share | {la}/{lb} |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for op in order:
        a, b = A.get(op), B.get(op)
        ratio = f"{a['worst_ms'] / b['worst_ms']:.2f}" if a and b and b.get("worst_ms") else ""
        print(f"| {op} | " + " | ".join(cells(a, lt_a) + cells(b, lt_b)) + f" | {ratio} |")
    for lab, lt, chips in ((la, lt_a, args.chips_a), (lb, lt_b, args.chips_b)):
        print(
            f"{lab}: layer {lt:.3f} ms x {chips} chips / {args.tokens} tok = {lt * 1e3 * chips / args.tokens:.2f} chip-us/token-layer"
        )


if __name__ == "__main__":
    main()
