# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-busy wall time per transformer-block iteration from a Tracy ops CSV, as the union of every op's device
firmware interval on each device -- the metric to compare configurations under FSDP, where the weight all-gathers run on
the CCL sub-device concurrently with the matmuls and a per-op sum (`block_profile_stats.py compare`'s "device only") counts
the overlap twice and can hide a gain or invent a loss.

    python models/tt_dit/tests/models/minimax_h3/tools/block_device_busy.py name=<ops_perf_results csv> [name=<csv> ...]

One iteration = the ops between two consecutive to_qkv AGMM calls (the last complete one per device); prints the mean, min
and max over devices of the union length and, for reference, the plain sum of kernel durations. Not a test.
"""
import collections
import csv
import statistics as st
import sys


def load(path):
    rows = [
        r
        for r in csv.DictReader(open(path))
        if r["GLOBAL CALL COUNT"].strip() and r["DEVICE KERNEL DURATION [ns]"].strip()
    ]
    for r in rows:
        r["_gcc"] = int(r["GLOBAL CALL COUNT"])
        r["_dur"] = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6
        r["_s"] = float(r["DEVICE FW START CYCLE"] or 0) / 1e6
        r["_e"] = float(r["DEVICE FW END CYCLE"] or 0) / 1e6
    return rows


def union_len(intervals):
    iv = sorted(intervals)
    total, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


def per_layer(path):
    bydev = collections.defaultdict(list)
    for r in load(path):
        bydev[int(r["DEVICE ID"])].append(r)
    union, total = [], []
    for rs in bydev.values():
        rs.sort(key=lambda r: r["_gcc"])
        agmm = [r for r in rs if "AllGatherMinimalMatmul" in r["OP CODE"]]
        if len(agmm) < 6:
            continue
        lo, hi = agmm[-6]["_gcc"], agmm[-3]["_gcc"]  # one period: to_qkv(i-1) .. to_qkv(i)
        ops = [r for r in rs if lo <= r["_gcc"] < hi]
        union.append(union_len([(r["_s"], r["_e"]) for r in ops]))
        total.append(sum(r["_dur"] for r in ops))
    return union, total


if __name__ == "__main__":
    print(f"{'config':24s} {'device-busy union, ms':>22s} {'min':>8s} {'max':>8s} {'sum of kernel durations':>24s}")
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        u, t = per_layer(path)
        print(f"{name:24s} {st.mean(u):22.2f} {min(u):8.2f} {max(u):8.2f} {st.mean(t):24.2f}")
