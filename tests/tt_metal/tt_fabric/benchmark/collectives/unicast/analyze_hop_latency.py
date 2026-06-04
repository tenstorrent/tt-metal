#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fit per-hop one-way fabric latency from a bench_unicast round-trip CSV.

The bench measures a src->dst->src round trip on the source chip's clock, so the
measured RTT traverses 2*hops fabric hops:

    rtt_ns(hops) = 2 * per_hop_ns * hops + const

Hence per_hop_one_way_ns = slope(rtt_ns_p50 vs hops) / 2. The intercept absorbs the
hop-independent constants (injection, far-end worker turnaround, wakeups) and is ignored.

Usage:
    python analyze_hop_latency.py hop_latency_2d.csv
"""
import csv
import sys


def main(path: str) -> int:
    hops, rtt = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                h = float(row["hops"])
                y = float(row["rtt_ns_p50"])
            except (KeyError, ValueError):
                continue
            if h > 0 and y > 0:
                hops.append(h)
                rtt.append(y)

    n = len(hops)
    if len(set(hops)) < 2:
        print(f"Need >=2 distinct hop counts to fit a line; got {n} rows, " f"{len(set(hops))} distinct hop values.")
        return 1

    # Ordinary least squares (no numpy dependency).
    mean_h = sum(hops) / n
    mean_y = sum(rtt) / n
    sxx = sum((h - mean_h) ** 2 for h in hops)
    sxy = sum((h - mean_h) * (y - mean_y) for h, y in zip(hops, rtt))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_h
    ss_tot = sum((y - mean_y) ** 2 for y in rtt)
    ss_res = sum((y - (slope * h + intercept)) ** 2 for h, y in zip(hops, rtt))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"samples              : {n}")
    print(f"hops                 : {sorted(set(int(h) for h in hops))}")
    print(f"RTT slope (ns/hop)   : {slope:.2f}")
    print(f"per-hop one-way (ns) : {slope / 2.0:.2f}")
    print(f"intercept (ns)       : {intercept:.2f}")
    print(f"R^2                  : {r2:.4f}")
    if r2 < 0.95:
        print("WARNING: R^2 < 0.95 — check for variance (pin links, raise --iters, " "confirm --trace-iters 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "hop_latency_2d.csv"))
