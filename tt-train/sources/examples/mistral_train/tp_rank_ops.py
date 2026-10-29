# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op ranked table of a TP Tracy profile for given chips (ms per profiled step, x10 for the 4-layer -> 40-layer
projection of the per-layer rows; the fixed rows stay as they are).

    python tp_rank_ops.py <ops_perf_results.csv> [--devices 0,2] [--warmup 2] [--top 28]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_tracy import classify, load, shape, split_steps  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--devices", default="0,2")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--top", type=int, default=28)
    args = p.parse_args()
    for dev in args.devices.split(","):
        rows = load(args.csv, dev)
        steps = split_steps(rows)[args.warmup :]
        n = len(steps)
        tot = collections.defaultdict(lambda: [0.0, 0.0])
        classes = collections.defaultdict(float)
        for s in steps:
            for r in s:
                us = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3
                key = (r["OP CODE"], shape(r, "INPUT_0"), shape(r, "INPUT_1"), r.get("CORE COUNT", ""))
                tot[key][0] += us / n
                tot[key][1] += 1 / n
                classes[classify(r["OP CODE"])] += us / n
        step = sum(v[0] for v in tot.values())
        print(
            f"chip {dev}: {n} steps, {step / 1e3:.2f} ms kernels per profiled step; classes: "
            + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in sorted(classes.items(), key=lambda t: -t[1]))
        )
        print(f"  {'ms/step':>8} {'ops':>5}  op | in0 | in1 | cores")
        for key, (us, cnt) in sorted(tot.items(), key=lambda t: -t[1][0])[: args.top]:
            print(
                f"  {us / 1e3:8.3f} {cnt:5.0f}  {key[0].replace('DeviceOperation', '')} | {key[1]} | {key[2]} | {key[3]}"
            )


if __name__ == "__main__":
    main()
