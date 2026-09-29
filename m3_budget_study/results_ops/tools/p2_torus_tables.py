#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2 tables: (4,2) carved rows 0-3 (1d, v1) vs (4,2) on the middle-rows 4x4 torus (T1 v1 ops, T2 v2 ops) vs (2,4).

  p2_torus_tables.py [--stat worst_ms|mean_ms]
Per op: mean over sparse layers 3-6 of the per-layer chip statistic (dense = layer 1). Markdown on stdout.
"""

import argparse
import csv
import os
from collections import defaultdict

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = [  # label, csv, run-id prefix
    ("carved v1", "per_op_4x2.csv", "p4x2"),
    ("torus v1 (T1)", "per_op_4x2_torus.csv", "p2t1"),
    ("torus v2 (T2)", "per_op_4x2_torus.csv", "p2t2"),
    ("torus v2-dispatch (T3)", "per_op_4x2_torus.csv", "p2t3"),
    ("[2,4]", "per_op.csv", "p0a"),
]
OPS = ["dispatch", "combine", "moe_reduce", "experts", "shared", "ag_kv", "layer_total"]
POINTS = [(W, h) for W in (4096, 8192) for h in (0, 141312, 548864)]


def load(stat):
    data = {}
    cache = {}
    for label, f, pre in CFGS:
        if f not in cache:
            cache[f] = list(csv.DictReader(open(os.path.join(R, f))))
        for W, h in POINTS:
            run = f"{pre}_w{W}_h{h}_prose"
            acc = defaultdict(list)
            for r in cache[f]:
                if r["run_id"] != run:
                    continue
                L = int(r["layer"])
                if L in (3, 4, 5, 6):
                    acc[r["op"]].append(float(r[stat]))
                elif L == 1:
                    acc["dense:" + r["op"]].append(float(r[stat]))
                acc["all:" + r["op"]].append(float(r[stat]))
            if acc:
                d = {k: sum(v) / len(v) for k, v in acc.items() if not k.startswith("all:")}
                d["stage7"] = sum(acc["all:layer_total"])  # layers 0-6 summed
                data[(label, W, h)] = d
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="worst_ms")
    a = ap.parse_args()
    D = load(a.stat)
    labels = [c[0] for c in CFGS if any((c[0], W, h) in D for W, h in POINTS)]
    for W, h in POINTS:
        print(f"\n**W={W}, h={h}** ({a.stat}, sparse ops = mean of layers 3-6)\n")
        print("| op | " + " | ".join(labels) + " |")
        print("|---|" + "---:|" * len(labels))
        rows = OPS + ["dense:layer_total", "stage7"]
        for op in rows:
            name = {
                "layer_total": "sparse layer",
                "dense:layer_total": "dense layer (1)",
                "stage7": "layers 0-6 sum",
            }.get(op, op)
            cells = []
            for lab in labels:
                v = D.get((lab, W, h), {}).get(op)
                cells.append(f"{v:.3f}" if v is not None else "")
            print(f"| {name} | " + " | ".join(cells) + " |")
        cells = []
        for lab in labels:
            v = D.get((lab, W, h), {}).get("layer_total")
            cells.append(f"{v * 1e3 * 8 / W:.2f}" if v else "")
        print("| sparse chip-us / token-layer | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
