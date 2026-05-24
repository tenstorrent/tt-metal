#!/usr/bin/env python3
"""
Compute per-zone duration statistics from a tt-metal device profiler CSV.

Pairs ZONE_START/ZONE_END for each (core, RISC, zone_name) in order of
appearance, computes durations in cycles, and reports
count / min / median / mean / std / p90 / p99 / max for each zone name.
"""
import argparse
import csv
from collections import defaultdict
from statistics import mean, pstdev


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="profile_log_device.csv")
    ap.add_argument("--freq-mhz", type=float, default=1350.0, help="chip frequency in MHz (for ns conversion)")
    ap.add_argument("--zones", nargs="+", default=None, help="Limit to these zone names")
    ap.add_argument("--per-core", action="store_true", help="Break down by (core_x, core_y, RISC) too")
    return ap.parse_args()


def main():
    args = parse_args()

    # Skip header line "ARCH: blackhole, ..."
    with open(args.csv) as f:
        first = f.readline()
        reader = csv.DictReader(f, skipinitialspace=True)
        durations = defaultdict(list)  # zone_name -> [cycles]
        per_core_durations = defaultdict(list)  # (zone, core) -> [cycles]
        # Pair START/END in order per (core_x, core_y, RISC, zone_name)
        open_starts = {}
        for row in reader:
            zname = row["zone name"].strip()
            ztype = row["type"].strip()
            if ztype not in ("ZONE_START", "ZONE_END"):
                continue
            if args.zones and zname not in args.zones:
                continue
            key = (row["core_x"].strip(), row["core_y"].strip(), row["RISC processor type"].strip(), zname)
            t = int(row["time[cycles since reset]"].strip())
            if ztype == "ZONE_START":
                open_starts.setdefault(key, []).append(t)
            else:
                stack = open_starts.get(key)
                if not stack:
                    continue
                start = stack.pop(0)
                dur = t - start
                durations[zname].append(dur)
                per_core_durations[(zname, key[0], key[1], key[2])].append(dur)

    def fmt_stats(vals, freq_mhz):
        if not vals:
            return "no data"
        v = sorted(vals)
        n = len(v)
        mn, mx = v[0], v[-1]
        med = v[n // 2]
        avg = mean(v)
        sd = pstdev(v)
        # quantile shortcuts
        p90 = v[int(0.90 * (n - 1))]
        p99 = v[int(0.99 * (n - 1))]
        ns = lambda c: c / freq_mhz * 1000.0  # cycles → ns
        return (
            f"n={n:5d}  "
            f"min={mn:4d} ({ns(mn):6.1f}ns)  "
            f"med={med:4d} ({ns(med):6.1f}ns)  "
            f"mean={avg:6.1f} ({ns(avg):6.1f}ns)  "
            f"std={sd:5.1f}  "
            f"p90={p90:4d} ({ns(p90):6.1f}ns)  "
            f"p99={p99:4d} ({ns(p99):6.1f}ns)  "
            f"max={mx:4d} ({ns(mx):6.1f}ns)"
        )

    print(f"# Frequency: {args.freq_mhz} MHz  (1 cycle = {1000.0/args.freq_mhz:.3f} ns)")
    print(f"# {first.strip()}")
    print()
    print("== Aggregated across all cores ==")
    for zname in sorted(durations.keys()):
        print(f"{zname:30s} {fmt_stats(durations[zname], args.freq_mhz)}")

    if args.per_core:
        print()
        print("== Per (zone, core_x, core_y, RISC) ==")
        for (zname, cx, cy, risc), vals in sorted(per_core_durations.items()):
            print(f"{zname:25s} ({cx:>2s},{cy:>2s},{risc:<8s}) {fmt_stats(vals, args.freq_mhz)}")


if __name__ == "__main__":
    main()
