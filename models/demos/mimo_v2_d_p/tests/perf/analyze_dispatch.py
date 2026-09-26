# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarise test_dispatch_perf.py: per case, the dispatch op's device time per chip and per RISC, joined with the
host-side routing stats (pairs received, remote share) and the implied fabric / write bandwidth.

    python models/demos/mimo_v2_d_p/tests/perf/analyze_dispatch.py <ops_perf_results.csv> [generated/mimo_dispatch/cases.jsonl]

Per chip, ``kernel`` is DEVICE KERNEL DURATION (first core start -> last core end); the RISC columns are the longest
per-core duration of that RISC: BRISC = writer (worker writer / sender fabric writer), NCRISC = worker reader,
TRISC = untilize compute, longest of TRISC0-2 (tile input only). ``GB/s`` = the busiest chip's received bytes / its kernel time.
"""

import collections
import csv
import json
import sys

COLS = {
    "kernel": "DEVICE KERNEL DURATION [ns]",
    "brisc": "DEVICE BRISC KERNEL DURATION [ns]",
    "ncrisc": "DEVICE NCRISC KERNEL DURATION [ns]",
}
TRISCS = [f"DEVICE TRISC{i} KERNEL DURATION [ns]" for i in range(3)]


def load(path):
    data = collections.defaultdict(lambda: collections.defaultdict(list))  # tag -> device -> [row per iteration]
    cur = None
    for x in csv.DictReader(open(path)):
        if x["OP TYPE"] == "signpost":
            c = x["OP CODE"]
            cur = c[: -len("_start")] if c.endswith("_start") else None
            continue
        if cur and "Dispatch" in x["OP CODE"]:
            row = {k: float(x[v] or 0) / 1e3 for k, v in COLS.items()}
            row["trisc"] = max(float(x[c] or 0) for c in TRISCS) / 1e3
            row["cores"] = int(x["CORE COUNT"] or 0)
            data[cur][int(x["DEVICE ID"])].append(row)
    return data


def main():
    csv_path = sys.argv[1]
    stats_path = sys.argv[2] if len(sys.argv) > 2 else "generated/mimo_dispatch/cases.jsonl"
    stats = {}
    try:
        for line in open(stats_path):
            s = json.loads(line)
            stats[s["tag"]] = s  # the last run of a tag wins
    except FileNotFoundError:
        pass
    hdr = f"{'case':<46s} {'kernel us':>18s} {'BRISC':>7s} {'NCRISC':>7s} {'TRISC':>7s} {'cores':>5s} {'pairs max':>9s} {'remote':>6s} {'GB/s':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for tag, devs in load(csv_path).items():
        mean = {d: {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]} for d, rows in devs.items()}
        kern = [m["kernel"] for m in mean.values()]
        worst = max(mean.values(), key=lambda m: m["kernel"])
        s = stats.get(tag)
        pairs = remote = gbps = ""
        if s:
            busiest = max(s["chips"].values(), key=lambda c: c["pairs"])
            tot = sum(c["pairs"] for c in s["chips"].values())
            pairs = str(busiest["pairs"])
            remote = f"{100 * sum(c['remote'] for c in s['chips'].values()) / max(tot, 1):.0f}%"
            gbps = f"{busiest['pairs'] * s['row_bytes'] / (max(kern) * 1e3):.1f}"
        span = f"{min(kern):.0f}-{max(kern):.0f}" if len(kern) > 1 else f"{kern[0]:.0f}"
        print(
            f"{tag:<46s} {span:>18s} {worst['brisc']:7.0f} {worst['ncrisc']:7.0f} {worst['trisc']:7.0f} {worst['cores']:5.0f} "
            f"{pairs:>9s} {remote:>6s} {gbps:>6s}"
        )


if __name__ == "__main__":
    main()
