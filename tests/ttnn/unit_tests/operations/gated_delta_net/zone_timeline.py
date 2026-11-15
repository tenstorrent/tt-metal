# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core zone timeline of one gated_delta_net launch from the device profiler log (not a test).

Run a `--profile` test with GDN_PROFILE_ZONES=1 (block zones) or 2 (also the point marks m_*), then:

    python3 tests/ttnn/unit_tests/operations/gated_delta_net/zone_timeline.py [--run N] [--group 12]

Times are us from the launch's earliest kernel start. Cores are listed in row-major NoC order, so with --group G the
k-th core of each block of G is position k of an R-BCAST key-head group (0 = q injector, 1 = k injector). Each zone
prints as start-end; a zone that runs several times prints its first start and last end.
"""

import argparse
import csv
from collections import defaultdict

FREQ_MHZ = 1350.0
LOG = "generated/profiler/.logs/profile_log_device.csv"
KERNEL_ZONES = ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL")


def load(path, run):
    with open(path) as f:
        f.readline()
        rows = list(csv.DictReader(f, skipinitialspace=True))
    runs = sorted({r["run host ID"] for r in rows})
    run_id = runs[run]
    return [r for r in rows if r["run host ID"] == run_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=LOG)
    parser.add_argument("--run", type=int, default=-1, help="launch index (default: last)")
    parser.add_argument("--group", type=int, default=0, help="cores per key-head group (R-BCAST)")
    parser.add_argument("--zones", default="", help="comma-separated zone names to show (default: all non-FW)")
    parser.add_argument("--marks", action="store_true", help="only point marks (zones named m_*), by time")
    args = parser.parse_args()
    rows = load(args.log, args.run)
    t0 = min(int(r["time[cycles since reset]"]) for r in rows if r["zone name"] in KERNEL_ZONES)
    spans = defaultdict(lambda: [None, None])  # (core, risc, zone) -> [first start, last end]
    for r in rows:
        if r["zone name"].endswith("-FW"):
            continue
        key = ((int(r["core_x"]), int(r["core_y"])), r["RISC processor type"], r["zone name"])
        t = (int(r["time[cycles since reset]"]) - t0) / FREQ_MHZ
        span = spans[key]
        if r["type"] == "ZONE_START":
            span[0] = t if span[0] is None else min(span[0], t)
        elif r["type"] == "ZONE_END":
            span[1] = t if span[1] is None else max(span[1], t)
    wanted = [z for z in args.zones.split(",") if z]
    cores = sorted({k[0] for k in spans}, key=lambda c: (c[1], c[0]))
    end = max(s[1] for s in spans.values() if s[1] is not None)
    print(f"launch end {end:.2f} us, {len(cores)} cores")
    for idx, core in enumerate(cores):
        pos = f"g{idx // args.group}.{idx % args.group}" if args.group else f"{idx}"
        parts = []
        for (c, risc, zone), (s, e) in sorted(spans.items(), key=lambda kv: (kv[1][0] or 0)):
            if c != core or (wanted and zone not in wanted) or (args.marks and not zone.startswith("m_")):
                continue
            if args.marks:
                parts.append(f"{risc[-1] if risc.startswith('TRISC') else risc[:2]}:{zone[2:]}@{e:.1f}")
                continue
            label = zone if zone not in KERNEL_ZONES else zone.split("-")[0]
            parts.append(f"{risc[:2]}:{label} {s:.1f}-{e:.1f}" if s is not None and e is not None else f"{label} ?")
        print(f"{pos:>6} {core}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
