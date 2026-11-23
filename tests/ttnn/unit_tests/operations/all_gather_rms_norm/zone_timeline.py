# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage timeline of all_gather_rms_norm from a device profile (profile_log_device.csv).

Usage: python tests/ttnn/unit_tests/operations/all_gather_rms_norm/zone_timeline.py [csv] [run_index]

For each chip, the run (host id) that contains this op's zones is taken (the last one by default); times are in
microseconds relative to the earliest kernel start of that run on that chip. For every zone name: earliest start,
latest end, and the median / max duration over the cores that recorded it (occupancy, not payload: a zone that
encloses a wait reports the wait).
"""

import collections
import csv
import statistics
import sys

OP_ZONES = {"ps_bulk", "pr_entry", "rd_x", "wr_own", "ps_fence_wait"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
    pick = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    with open(path) as f:
        first = f.readline()
        freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader((line.replace(", ", ",") for line in f)))
    by_dev = collections.defaultdict(list)
    for r in rows:
        by_dev[r["PCIe slot"]].append(r)
    for dev, rs in sorted(by_dev.items()):
        runs = sorted({r["run host ID"] for r in rs if r["zone name"] in OP_ZONES}, key=int)
        if not runs:
            continue
        run = runs[pick]
        rr = [r for r in rs if r["run host ID"] == run]
        starts = {}
        spans = collections.defaultdict(list)
        t0 = min(int(r["time[cycles since reset]"]) for r in rr if r["zone name"].endswith("-KERNEL"))
        t_end = max(int(r["time[cycles since reset]"]) for r in rr if r["zone name"].endswith("-KERNEL"))
        for r in rr:
            key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                starts[key] = t
            elif r["type"] == "ZONE_END" and key in starts:
                spans[r["zone name"]].append((starts.pop(key) - t0, t - t0, key[:3]))
        us = lambda c: c / freq
        print(f"=== chip slot {dev}, run {run}: kernel span {us(t_end - t0):.2f} us")
        order = sorted(spans.items(), key=lambda kv: min(s for s, _, _ in kv[1]))
        for name, sp in order:
            durs = [e - s for s, e, _ in sp]
            print(
                f"  {name:22s} n={len(sp):4d} start {us(min(s for s, _, _ in sp)):8.2f}  end {us(max(e for _, e, _ in sp)):8.2f}"
                f"  dur med {us(statistics.median(durs)):7.2f} max {us(max(durs)):7.2f}"
            )


if __name__ == "__main__":
    main()
