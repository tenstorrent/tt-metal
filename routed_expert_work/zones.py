# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Per-stage zone report for moe_fused_swiglu from generated/profiler/.logs/profile_log_device.csv
(captured with MOE_FUSED_SWIGLU_STAGE_PROFILE=1 TT_METAL_DEVICE_PROFILER=1).

For every (RISC, zone) prints, across cores: median/min/max of the per-core TOTAL time spent in that
zone (a zone entered several times on one core is summed), the median number of entries, and the
median start offset of the FIRST entry relative to that core's *-KERNEL zone start on the same RISC.
All times in microseconds at the CSV's CHIP_FREQ.

Usage: python routed_expert_work/zones.py [csv] [--risc TRISC_0] [--filter substr] [--per-core zone]
"""
import argparse
import csv
import statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        head = f.readline()
        freq_mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        rows = []
        for r in reader:
            if len(r) < 12:
                continue
            rows.append(
                {
                    "core": (int(r[1]), int(r[2])),
                    "risc": r[3].strip(),
                    "t": int(r[5]),
                    "zone": r[10].strip(),
                    "type": r[11].strip(),
                }
            )
    return freq_mhz, rows


def spans(rows):
    """-> {(core, risc): {zone: [(start, end), ...]}} pairing ZONE_START/ZONE_END in order."""
    out = defaultdict(lambda: defaultdict(list))
    open_ = defaultdict(list)
    for r in rows:
        key = (r["core"], r["risc"], r["zone"])
        if r["type"] == "ZONE_START":
            open_[key].append(r["t"])
        elif r["type"] == "ZONE_END":
            if open_[key]:
                s = open_[key].pop()
                out[(r["core"], r["risc"])][r["zone"]].append((s, r["t"]))
    return out


def report(path, risc_filter=None, name_filter=None, per_core_zone=None):
    freq, rows = load(path)
    sp = spans(rows)
    us = lambda cyc: cyc / freq  # noqa: E731

    # kernel start per (core, risc)
    kstart = {}
    kdur = {}
    for (core, risc), zones in sp.items():
        for z, lst in zones.items():
            if z.endswith("-KERNEL"):
                kstart[(core, risc)] = lst[0][0]
                kdur[(core, risc)] = lst[0][1] - lst[0][0]

    agg = defaultdict(lambda: {"tot": [], "n": [], "off": [], "first_dur": []})
    for (core, risc), zones in sp.items():
        if risc_filter and risc != risc_filter:
            continue
        for z, lst in zones.items():
            if name_filter and name_filter not in z:
                continue
            tot = sum(e - s for s, e in lst)
            a = agg[(risc, z)]
            a["tot"].append(us(tot))
            a["n"].append(len(lst))
            if (core, risc) in kstart:
                a["off"].append(us(lst[0][0] - kstart[(core, risc)]))
            a["first_dur"].append(us(lst[0][1] - lst[0][0]))

    print(f"clock {freq:.0f} MHz; cores with data: {len({c for c, _ in sp})}")
    print(f"{'risc':<8} {'zone':<30} {'cores':>5} {'n':>3} {'off_med':>8} {'med_us':>8} {'min_us':>8} {'max_us':>8}")
    for (risc, z), a in sorted(
        agg.items(), key=lambda kv: (kv[0][0], statistics.median(kv[1]["off"]) if kv[1]["off"] else 0)
    ):
        off = statistics.median(a["off"]) if a["off"] else float("nan")
        print(
            f"{risc:<8} {z:<30} {len(a['tot']):>5} {int(statistics.median(a['n'])):>3} {off:8.1f} "
            f"{statistics.median(a['tot']):8.1f} {min(a['tot']):8.1f} {max(a['tot']):8.1f}"
        )

    if per_core_zone:
        print(f"\nper-core totals for zone {per_core_zone}:")
        for (core, risc), zones in sorted(sp.items()):
            if per_core_zone in zones and (not risc_filter or risc == risc_filter):
                tot = sum(e - s for s, e in zones[per_core_zone])
                print(f"  core {core} {risc}: {us(tot):8.1f} us over {len(zones[per_core_zone])} entries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--risc", default=None)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--per-core", default=None)
    a = ap.parse_args()
    report(a.csv, a.risc, a.filter, a.per_core)


def timeline(path, core, riscs=("NCRISC", "BRISC", "TRISC_0")):
    """Print every zone entry of one core (x,y) with start/end offsets (us) from that RISC's kernel start."""
    freq, rows = load(path)
    sp = spans(rows)
    for risc in riscs:
        zones = sp.get((core, risc))
        if not zones:
            continue
        k0 = [lst[0][0] for z, lst in zones.items() if z.endswith("-KERNEL")][0]
        ev = []
        for z, lst in zones.items():
            if z.endswith("-FW") or z.endswith("-KERNEL"):
                continue
            for s, e in lst:
                ev.append(((s - k0) / freq, (e - k0) / freq, z))
        ev.sort()
        print(f"--- core {core} {risc}")
        for s, e, z in ev:
            print(f"  {s:7.1f} -> {e:7.1f}  ({e - s:6.1f})  {z}")
