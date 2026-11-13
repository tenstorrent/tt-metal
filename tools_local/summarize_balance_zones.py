#!/usr/bin/env python3
# Local-only helper (standard library only, profile branch): per-device and per-core busy time of the sparse_ring_sdpa
# load-balancing profile zones (B- balance kernel, R- SDPA reader, C- compute, W- SDPA writer, G- gather writer) in
# the LAST op call per device of a device profiler CSV ($TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv).
#
# A core's busy time in a zone = the sum of its instances (R-KVWAIT runs once per ring iteration). Per device and
# zone: max / mean over the cores that recorded it; the "all" row: max / mean over every core of every device.
# "span" = the device's first to last marker of the call. Times in us. Compute zones: one TRISC (--trisc).
#
# Usage: python3 tools_local/summarize_balance_zones.py <profile_log_device.csv> [--call -1] [--trisc TRISC_1]

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict

PREFIXES = ("B-", "R-", "C-", "W-", "G-")
ORDER = [
    "R-COST",
    "B-COSTS",
    "B-EXCHANGE",
    "B-ASSIGN",
    "B-TABLES",
    "R-RANGE",
    "B-IMPORTS",
    "R-IMPORT",
    "R-RING",
    "R-KVWAIT",
    "C-RING",
    "W-OUT",
    "B-OUTPUTS",
    "G-BALANCE",
    "B-RETURNS",
]


def load(path):
    with open(path, newline="") as f:
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline())
        freq = float(m.group(1)) if m else 1350.0
        reader = csv.reader(f, skipinitialspace=True)
        cols = [c.strip() for c in next(reader)]
        ix = {c: i for i, c in enumerate(cols)}
        rows = []
        for r in reader:
            if len(r) < len(cols) - 1:
                continue
            typ = r[ix["type"]].strip()
            if typ not in ("ZONE_START", "ZONE_END"):
                continue
            tc = r[ix["trace id counter"]].strip()
            rows.append(
                (
                    int(r[ix["PCIe slot"]]),
                    (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                    r[ix["RISC processor type"]].strip(),
                    int(r[ix["time[cycles since reset]"]]),
                    (r[ix["run host ID"]].strip(), int(tc) if tc else -1),
                    r[ix["zone name"]].strip(),
                    typ == "ZONE_START",
                )
            )
    return rows, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--call", type=int, default=-1, help="which call (by start time) among those with the zones")
    ap.add_argument("--trisc", default="TRISC_1", help="the TRISC whose compute zones count")
    args = ap.parse_args()

    rows, freq = load(args.csv)
    first = defaultdict(dict)  # chip -> call -> first marker
    for chip, core, risc, t, call, zone, st in rows:
        if zone.startswith(PREFIXES):
            first[chip][call] = min(first[chip].get(call, t), t)
    if not first:
        sys.exit("no B-/R-/C-/W-/G- zones in the CSV (profile branch build? TT_METAL_DEVICE_PROFILER=1? right file?)")
    picked = {chip: sorted(d, key=d.get)[args.call] for chip, d in first.items()}
    rows = sorted((r for r in rows if picked.get(r[0]) == r[4]), key=lambda r: r[3])
    t0, t1 = {}, {}
    for r in rows:
        t0[r[0]] = min(t0.get(r[0], r[3]), r[3])
        t1[r[0]] = max(t1.get(r[0], r[3]), r[3])

    busy = defaultdict(lambda: defaultdict(float))  # (zone) -> (chip, core) -> us
    stacks = defaultdict(list)
    for chip, core, risc, t, call, zone, st in rows:
        if not zone.startswith(PREFIXES) or (zone.startswith("C-") and risc != args.trisc):
            continue
        key = (chip, core, risc, zone)
        if st:
            stacks[key].append(t)
        elif stacks[key]:
            busy[zone][(chip, core)] += (t - stacks[key].pop()) / freq

    def us(v):
        return f"{v:.1f}" if v < 100 else f"{v:.0f}"

    zones = sorted(busy, key=lambda z: (ORDER.index(z) if z in ORDER else len(ORDER), z))
    chips = sorted(t0)
    w = 15
    print(f"{'device':<8}{'span':>8}" + "".join(f"{z:>{w}}" for z in zones))
    print(f"{'':<16}" + "".join(f"{'max/mean':>{w}}" for _ in zones))
    for chip in chips:
        cells = []
        for z in zones:
            v = [us for (c, _), us in busy[z].items() if c == chip]
            cells.append(f"{us(max(v))}/{us(statistics.mean(v))}" if v else "-")
        print(f"{chip:<8}{(t1[chip] - t0[chip]) / freq:>8.0f}" + "".join(f"{c:>{w}}" for c in cells))
    spans = [(t1[c] - t0[c]) / freq for c in chips]
    cells = []
    for z in zones:
        v = list(busy[z].values())
        cells.append(f"{us(max(v))}/{us(statistics.mean(v))}")
    print(f"{'all':<8}{max(spans):>8.0f}" + "".join(f"{c:>{w}}" for c in cells))
    print(f"(us at {freq:.0f} MHz; per device: max / mean over its cores of each core's busy time in the zone)")


if __name__ == "__main__":
    main()
