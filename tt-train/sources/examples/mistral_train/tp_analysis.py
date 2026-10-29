# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-chip view of a TP Tracy ``ops_perf_results*.csv``: kernel time, collectives, idle, and whether one chip runs
its matmuls slower than the others (a throttled chip stalls the mesh in every collective).

    python tp_analysis.py --warmup 2 evidence/<tag>_tracy_tp4_4layers/reports/*/ops_perf_results*.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_tracy import load, split_steps, shape  # noqa: E402

csv.field_size_limit(10**9)
COLLECTIVE = ("ReduceScatter", "AllGather", "AllReduce", "AllBroadcast")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--devices", default="0,1,2,3")
    args = p.parse_args()
    devs = args.devices.split(",")
    per = {}
    print("chip | ops | kernels ms | matmul ms | collectives ms | span ms (cycles/1.35 GHz) | idle % | eff. clock GHz")
    for dev in devs:
        steps = split_steps(load(args.csv, device=dev))
        if len(steps) <= args.warmup + 1:
            print(f"{dev:4s} | only {len(steps)} steps")
            continue
        s, nxt = steps[args.warmup], steps[args.warmup + 1]
        kern = sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in s) / 1e6
        mm = sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in s if "Matmul" in r["OP CODE"]) / 1e6
        coll = (
            sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in s if any(c in r["OP CODE"] for c in COLLECTIVE)) / 1e6
        )
        c0, c1 = int(s[0]["DEVICE FW START CYCLE"]), int(nxt[0]["DEVICE FW START CYCLE"])
        h0, h1 = int(s[0]["HOST START TS"]), int(nxt[0]["HOST START TS"])
        span = (c1 - c0) / 1.35e6
        print(
            f"{dev:4s} | {len(s):3d} | {kern:10.1f} | {mm:9.1f} | {coll:14.1f} | {span:8.1f} | {100 * (1 - kern / span):6.1f} | {(c1 - c0) / (h1 - h0):.3f}"
        )
        by = collections.defaultdict(float)
        for r in s:
            if "Matmul" in r["OP CODE"]:
                by[(shape(r, "INPUT_0"), shape(r, "INPUT_1"))] += float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6
        per[dev] = by
    if len(per) > 1:
        ref = devs[0]
        print(f"\nmatmul time relative to chip {ref}, top shapes:")
        for k in sorted(per[ref], key=lambda k: -per[ref][k])[:6]:
            print(
                f"  {per[ref][k]:7.2f} ms  "
                + "  ".join(f"chip{d} x{per[d].get(k, 0) / per[ref][k]:.2f}" for d in devs if d in per)
                + f"  {k}"
            )
        coll_by = collections.defaultdict(lambda: [0.0, 0])
        s = split_steps(load(args.csv, device=ref))[args.warmup]
        for r in s:
            if any(c in r["OP CODE"] for c in COLLECTIVE):
                k = (r["OP CODE"].replace("DeviceOperation", ""), shape(r, "INPUT_0"))
                coll_by[k][0] += float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3
                coll_by[k][1] += 1
        print(f"\ncollectives on chip {ref}:")
        for k, (us, n) in sorted(coll_by.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"  {us / 1e3:7.2f} ms {n:4d} x {us / n:7.0f} us  {k}")


if __name__ == "__main__":
    main()
