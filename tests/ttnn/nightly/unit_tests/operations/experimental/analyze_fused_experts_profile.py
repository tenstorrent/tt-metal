# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize the fused_experts kernel-profiler zones from profile_log_device.csv.

Pairs ZONE_START/ZONE_END markers per (core, RISC, zone, run) and reports, for each zone,
the min / mean / max duration across cores in microseconds -- which is what separates the
gate_up phase, the gather/broadcast sync and the down phase in the reader loop.
"""

import collections
import csv
import statistics
import sys

CHIP_FREQ_MHZ = 1350.0


def load(path):
    rows = list(csv.reader(open(path)))
    hdr = next(i for i, r in enumerate(rows) if r and "PCIe slot" in r[0])
    cols = [c.strip() for c in rows[hdr]]
    idx = {c: i for i, c in enumerate(cols)}
    return [r for r in rows[hdr + 1 :] if len(r) >= len(cols)], idx


def main(path="generated/profiler/.logs/profile_log_device.csv"):
    data, idx = load(path)
    # (run, core, risc, zone) -> {"start": t, "dur": [..]}
    opens = collections.defaultdict(list)
    durations = collections.defaultdict(list)  # (zone, risc) -> [cycles]
    per_run = collections.defaultdict(lambda: collections.defaultdict(list))

    for r in data:
        zone = r[idx["zone name"]].strip()
        typ = r[idx["type"]].strip()
        core = (r[idx["core_x"]].strip(), r[idx["core_y"]].strip())
        risc = r[idx["RISC processor type"]].strip()
        run = r[idx["run host ID"]].strip()
        t = int(r[idx["time[cycles since reset]"]])
        key = (run, core, risc, zone)
        if typ == "ZONE_START":
            opens[key].append(t)
        elif typ == "ZONE_END" and opens[key]:
            dt = t - opens[key].pop()
            durations[(zone, risc)].append(dt)
            per_run[run][(zone, risc)].append(dt)

    runs = sorted(per_run)
    print(f"{len(data)} marker rows, {len(runs)} runs; showing last run ({runs[-1]})\n")
    last = per_run[runs[-1]]
    print(f"{'zone':<26} {'risc':<8} {'n':>4} {'min us':>9} {'mean us':>9} {'max us':>9}")
    print("-" * 70)
    for (zone, risc), vals in sorted(last.items(), key=lambda kv: -statistics.mean(kv[1])):
        us = [v / CHIP_FREQ_MHZ for v in vals]
        print(f"{zone:<26} {risc:<8} {len(us):>4} {min(us):>9.1f} " f"{statistics.mean(us):>9.1f} {max(us):>9.1f}")

    # Per-core grid for the DRAM-read zones: the op runs on logical cores (0,0)..(7,7) and
    # each core's DRAM shard id is its flat index y*8+x, so the bank it reads is (y*8+x) % 8
    # == x. A column-wise pattern here means NoC routing / bank contention, not compute.
    per_core = collections.defaultdict(dict)  # zone -> (x,y) -> us
    for r in data:
        if r[idx["run host ID"]].strip() != runs[-1] or r[idx["type"]].strip() != "ZONE_END":
            continue
        zone = r[idx["zone name"]].strip()
        if not zone.startswith("FE_PHASE"):
            continue
    # Rebuild with pairing (ZONE_END alone has no duration), reusing the same walk as above.
    opens2 = collections.defaultdict(list)
    for r in data:
        if r[idx["run host ID"]].strip() != runs[-1]:
            continue
        zone = r[idx["zone name"]].strip()
        if not zone.startswith("FE_"):
            continue
        core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
        typ = r[idx["type"]].strip()
        t = int(r[idx["time[cycles since reset]"]])
        key = (core, zone)
        if typ == "ZONE_START":
            opens2[key].append(t)
        elif typ == "ZONE_END" and opens2[key]:
            per_core[zone][core] = (t - opens2[key].pop()) / CHIP_FREQ_MHZ

    xs = sorted({c[0] for z in per_core.values() for c in z})
    ys = sorted({c[1] for z in per_core.values() for c in z})
    for zone in ("FE_PHASE1_GATE_UP", "FE_PHASE2_DOWN"):
        if zone not in per_core:
            continue
        print(f"\n{zone} per core (us), rows = logical y, cols = logical x:")
        print("      " + "".join(f"{x:>8}" for x in xs))
        for y in ys:
            cells = "".join(f"{per_core[zone].get((x, y), float('nan')):>8.1f}" for x in xs)
            print(f"y={y:<3} {cells}")


if __name__ == "__main__":
    main(*sys.argv[1:])
