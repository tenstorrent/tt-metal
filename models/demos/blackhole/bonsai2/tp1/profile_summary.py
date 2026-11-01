# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device kernel time per decode step for each signposted region of a tracy ops_perf_results CSV.

    python models/demos/blackhole/bonsai2/tp1/profile_summary.py <ops_perf_results_*.csv> [--steps 4] [--sequence]

Ops without a device kernel (views, host-side reshapes) are not counted. --sequence also prints the ops of one
step of each region in order, with shapes, layouts and core counts.
"""
import argparse
import collections
import csv


def _shape(row, t):
    dims = [(row.get(f"{t}_{d}_PAD[LOGICAL]") or "").strip() for d in "WZYX"]
    if not any(dims):
        return "-"
    return "x".join(d.split("[")[1].rstrip("]") if "[" in d else d for d in dims)


def _tensor(row, t):
    layout = (row.get(f"{t}_LAYOUT") or "").replace("Layout::", "")
    mem = (row.get(f"{t}_MEMORY") or "").replace("DEV_0_", "").replace("TensorMemoryLayout::", "")
    dtype = (row.get(f"{t}_DATATYPE") or "").replace("DataType::", "")
    return f"{_shape(row, t)} {layout} {dtype} {mem}".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--steps", type=int, default=4, help="decode steps inside each signposted region")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--sequence", action="store_true", help="print one step's ops in order")
    a = p.parse_args()

    regions = collections.OrderedDict()
    cur = None
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                cur = row["OP CODE"]
                regions.setdefault(cur, [])
                continue
            ns = (row.get("DEVICE KERNEL DURATION [ns]") or "").strip()
            if cur is None or ns in ("", "-"):
                continue
            row["_ns"] = float(ns)
            regions[cur].append(row)

    for name, ops in regions.items():
        if not ops:
            continue
        total_us = sum(r["_ns"] for r in ops) / a.steps / 1e3
        print(f"== {name}: {len(ops) / a.steps:.1f} device ops/step, {total_us:.1f} us device kernel time/step")
        by_op = collections.defaultdict(lambda: [0, 0.0])
        for r in ops:
            by_op[r["OP CODE"]][0] += 1
            by_op[r["OP CODE"]][1] += r["_ns"]
        for op, (n, ns) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[: a.top]:
            print(f"   {op:<56} {n / a.steps:>5.1f}/step {ns / a.steps / 1e3:>8.1f} us")
        if a.sequence:
            per_step = len(ops) // a.steps
            for i, r in enumerate(ops[:per_step]):
                print(
                    f"   {i:>3} {r['OP CODE']:<44} {r['_ns'] / 1e3:>7.1f} us  cores={r.get('CORE COUNT', ''):>3}"
                    f"  in0={_tensor(r, 'INPUT_0')}  ->  out0={_tensor(r, 'OUTPUT_0')}"
                )


if __name__ == "__main__":
    main()
