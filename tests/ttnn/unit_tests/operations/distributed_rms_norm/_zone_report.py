# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone report for distributed_rms_norm perf runs (DRMS_PERF_ZONES=1), read off profile_log_device.csv.

For the last N runs of every chip (default: the last run): per (zone, RISC) the per-core total (max / mean over
cores), first start and last end (relative to the earliest kernel start of that run on that chip, max over cores),
and the per-RISC kernel end distribution. Occupancy, not payload: a zone that waits counts its wait.

    python3 tests/ttnn/unit_tests/operations/distributed_rms_norm/_zone_report.py [--log <csv>] [--runs 1] [--chip N]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--run-index", type=int, default=-1, help="which run per chip (python index over sorted run ids)")
    ap.add_argument("--chip", type=int, default=None)
    args = ap.parse_args()
    with open(args.log) as fh:
        header = fh.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    ns = lambda c: c * 1000.0 / mhz
    by = defaultdict(list)
    for r in rows:
        by[(int(r["PCIe slot"]), int(r["run host ID"]))].append(r)
    chips = sorted({c for c, _ in by})
    for chip in chips:
        if args.chip is not None and chip != args.chip:
            continue
        runs = sorted(rid for c, rid in by if c == chip)
        rid = runs[args.run_index]
        rr = by[(chip, rid)]
        stack = defaultdict(list)
        spans = defaultdict(list)  # (zone, risc) -> list of (core, start, end)
        for r in rr:
            key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                stack[key].append(t)
            elif r["type"] == "ZONE_END" and stack[key]:
                s = stack[key].pop()
                spans[(r["zone name"], r["RISC processor type"])].append(((r["core_x"], r["core_y"]), s, t))
        kstarts = [s for (z, _), v in spans.items() if z.endswith("-KERNEL") for _, s, _ in v]
        if not kstarts:
            continue
        t0 = min(kstarts)
        print(f"\n=== chip {chip} run {rid} ({len(runs)} runs) ===")
        print(
            f"{'zone':24s} {'risc':7s} {'cores':>5s} {'tot max':>9s} {'tot mean':>9s} {'first':>8s} {'end max':>8s} {'end med':>8s}"
        )
        for (z, risc), v in sorted(spans.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if z.endswith("-FW"):
                continue
            tot = defaultdict(float)
            last = defaultdict(int)
            for core, s, e in v:
                tot[core] += e - s
                last[core] = max(last[core], e)
            tots = list(tot.values())
            ends = [ns(e - t0) for e in last.values()]
            first = ns(min(s for _, s, _ in v) - t0)
            print(
                f"{z:24s} {risc:7s} {len(tot):5d} {ns(max(tots)):9.0f} {ns(statistics.mean(tots)):9.0f} "
                f"{first:8.0f} {max(ends):8.0f} {statistics.median(ends):8.0f}"
            )


if __name__ == "__main__":
    main()
