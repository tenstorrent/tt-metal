# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone report for matmul_reduce_scatter (reads generated/profiler/.logs/profile_log_device.csv).

Usage: python tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_zone_report.py [csv] [--device D] [--run R]
Picks the LAST program run per device (the last traced replay) unless --run gives a run host ID / trace counter.
For every (zone, RISC) prints: cores, executions per core, mean per-exec us, p50/max of per-core sum, and the
mean start / end offset (us) of each occurrence index relative to the program start (the timeline).
Also checks the marker budget (markers per (core, RISC)) and kernel-span coverage.
"""
import csv
import sys
from collections import defaultdict
from statistics import median

FREQ_MHZ = 1350.0


def main():
    args = sys.argv[1:]
    path = "generated/profiler/.logs/profile_log_device.csv"
    dev_filter = None
    if args and not args[0].startswith("--"):
        path = args.pop(0)
    if "--device" in args:
        dev_filter = int(args[args.index("--device") + 1])
    with open(path) as f:
        first = f.readline()
        if "CHIP_FREQ" in first:
            global FREQ_MHZ
            FREQ_MHZ = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    ix = {h: i for i, h in enumerate(hdr)}
    by_dev = defaultdict(list)
    for r in rows[1:]:
        r = [c.strip() for c in r]
        dev = int(r[ix["PCIe slot"]])
        by_dev[dev].append(r)
    for dev in sorted(by_dev):
        if dev_filter is not None and dev != dev_filter:
            continue
        rs = by_dev[dev]
        # run key: (run host ID, trace id counter)
        keyf = lambda r: (r[ix["run host ID"]], r[ix["trace id counter"]])
        keys = []
        for r in rs:
            k = keyf(r)
            if not keys or keys[-1] != k:
                if k not in keys:
                    keys.append(k)
        last = keys[-1]
        rs = [r for r in rs if keyf(r) == last]
        # stacks per (core, risc, zone)
        open_ = {}
        occ = defaultdict(list)  # (zone, risc) -> list of (core, idx, start, end)
        markers = defaultdict(int)
        idx_ctr = defaultdict(int)
        t0 = min(int(r[ix["time[cycles since reset]"]]) for r in rs)
        for r in rs:
            core = (r[ix["core_x"]], r[ix["core_y"]])
            risc = r[ix["RISC processor type"]]
            z = r[ix["zone name"]]
            t = int(r[ix["time[cycles since reset]"]])
            typ = r[ix["type"]]
            if typ not in ("ZONE_START", "ZONE_END"):
                continue
            markers[(core, risc)] += 1
            key = (core, risc, z)
            if typ == "ZONE_START":
                open_.setdefault(key, []).append(t)
            else:
                if open_.get(key):
                    s = open_[key].pop()
                    i = idx_ctr[key]
                    idx_ctr[key] += 1
                    occ[(z, risc)].append((core, i, s - t0, t - t0))
        print(f"=== device {dev} run {last}  (us at {FREQ_MHZ} MHz)")
        us = lambda c: c / FREQ_MHZ
        for z, risc in sorted(occ, key=lambda k: (k[1], k[0])):
            L = occ[(z, risc)]
            cores = sorted({c for c, _, _, _ in L})
            per_core = defaultdict(float)
            for c, _, s, e in L:
                per_core[c] += e - s
            sums = list(per_core.values())
            nexec = len(L) / len(cores)
            durs = [e - s for _, _, s, e in L]
            line = (
                f"{z:18s} {risc:7s} cores={len(cores):3d} exec/core={nexec:5.1f} mean={us(sum(durs)/len(durs)):8.2f}"
                f" sum p50={us(median(sums)):8.2f} max={us(max(sums)):8.2f}"
            )
            print(line)
            if (
                "--all" in args
                or z in ("MM", "FIN", "ARR_WAIT", "PORT_GATHER_UNIT", "SEND_UNIT", "WR_BARRIER", "ACK_WAIT")
                or z.endswith("KERNEL")
            ):
                byi = defaultdict(list)
                for _, i, s, e in L:
                    byi[i].append((s, e))
                tl = "  ".join(
                    f"#{i}:[{us(median(s for s,_ in v)):.1f}-{us(median(e for _,e in v)):.1f}|max end {us(max(e for _,e in v)):.1f}]"
                    for i, v in sorted(byi.items())
                )
                print("      " + tl)
        mx = max(markers.values())
        print(f"  max markers per (core,risc): {mx}  (cap ~254 incl. fw)")


if __name__ == "__main__":
    main()
