# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a traced test_matmul_reduce_scatter_perf.py --profile run: per case (in collection order), the op latency of
each traced replay = MAX over the group's devices of DEVICE KERNEL DURATION; prints the replays and their median (us).

Usage: python tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_summary.py <ops_perf_results.csv> [n_devices=4]
"""
import csv
import sys
from statistics import median


def main():
    path = sys.argv[1]
    ndev = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = list(csv.reader(open(path)))
    ix = {h: i for i, h in enumerate(rows[0])}
    cases, cur, prev_generic = [], [], False
    for r in rows[1:]:
        op = r[ix["OP CODE"]]
        if op.startswith("GenericOp"):
            prev_generic = True
            d = r[ix["DEVICE KERNEL DURATION [ns]"]]
            if d:
                cur.append(int(d))
        elif prev_generic:  # an input-prep op after the op's calls: next case
            cases.append(cur)
            cur, prev_generic = [], False
    if cur:
        cases.append(cur)
    for i, c in enumerate(cases):
        reps = [max(c[j : j + ndev]) / 1000 for j in range(len(c) - 3 * ndev, len(c), ndev)]
        print(f"case {i}: replays {' '.join(f'{x:.1f}' for x in reps)}  median {median(reps):.1f} us")


if __name__ == "__main__":
    main()
