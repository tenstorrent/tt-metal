# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Summarize a tracy ops_perf_results CSV of profile_step: per op code, device kernel time vs op-to-op gap.

    python -m models.demos.inkling2_mini.demo.profile_report <ops_perf_results_*.csv> [--top 25]

Uses the last trace replay session in the file (one decode step)."""

import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    sess_key = next((k for k in rows[0] if "TRACE REPLAY SESSION" in k), None)
    if sess_key:
        sessions = sorted({r[sess_key] for r in rows if r[sess_key] not in ("", None)}, key=lambda s: int(float(s)))
        if sessions:
            rows = [r for r in rows if r[sess_key] == sessions[-1]]

    def num(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    kern = "DEVICE KERNEL DURATION [ns]"
    gap = "OP TO OP LATENCY [ns]"

    def shape(r, i):
        dims = [r.get(f"INPUT_{i}_{d}_PAD[LOGICAL]", "") for d in "WZYX"]
        dims = [d.split("[")[-1].rstrip("]") if "[" in d else d for d in dims]
        return "x".join(d for d in dims if d) + ("/" + r.get(f"INPUT_{i}_DATATYPE", "")[:6] if dims[0] else "")

    gaps = [num(r, gap) for r in rows[1:]]  # first op's gap = idle time before the replay
    n = len(rows)
    tot_k = sum(num(r, kern) for r in rows)
    print(
        f"ops {n}  device kernel {tot_k / 1e6:.2f} ms  op-to-op gaps {sum(gaps) / 1e6:.2f} ms "
        f"(median {sorted(gaps)[len(gaps) // 2] / 1e3:.2f} us)  span {(tot_k + sum(gaps)) / 1e6:.2f} ms"
    )
    for title, keyf in (
        ("op code", lambda r: r["OP CODE"].replace("DeviceOperation", "")),
        (
            "op code + input shapes",
            lambda r: f"{r['OP CODE'].replace('DeviceOperation', '')} {shape(r, 0)} {shape(r, 1)}",
        ),
    ):
        agg = defaultdict(lambda: [0, 0.0, 0.0])
        for i, r in enumerate(rows):
            g = agg[keyf(r)]
            g[0] += 1
            g[1] += num(r, kern)
            g[2] += num(r, gap) if i else 0.0
        print(f"\n| {title} | count | kernel ms | gap ms | kernel us/op |")
        print("|---|---|---|---|---|")
        for op, (c, k, g) in sorted(agg.items(), key=lambda x: -x[1][1])[: a.top]:
            print(f"| {op} | {c} | {k / 1e6:.2f} | {g / 1e6:.2f} | {k / c / 1e3:.1f} |")


if __name__ == "__main__":
    main()
