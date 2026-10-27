"""Summarize halo probe zones from a device profiler CSV: per device and call, times relative to the device's
first kernel start, in microseconds. Usage: python parse_halo_prof.py <profile_log_device.csv> <calls>"""

import collections
import csv
import statistics
import sys

path, calls = sys.argv[1], int(sys.argv[2])
MHZ = 1350.0
ZONES = ["HR_READ", "HW_OPEN", "HW_SEND", "HW_CLOSE", "HR_ARRIVE", "SDPA_HALO_WAIT"]

# (dev, run) -> zone -> list of (core, start, end)
starts = {}
data = collections.defaultdict(lambda: collections.defaultdict(list))
kernel = collections.defaultdict(lambda: [float("inf"), 0])
with open(path) as f:
    next(f)
    rows = csv.reader(f)
    next(rows)
    for r in rows:
        dev, cx, cy, risc, _tid, t, _d, run = r[0].strip(), r[1], r[2], r[3], r[4], int(r[5]), r[6], r[7].strip()
        zone, kind = r[10].strip(), r[11].strip()
        key = (dev, run)
        if zone.endswith("-KERNEL"):
            if kind == "ZONE_START":
                kernel[key][0] = min(kernel[key][0], t)
            elif kind == "ZONE_END":
                kernel[key][1] = max(kernel[key][1], t)
        elif zone in ZONES:
            sk = (key, zone, cx, cy, risc)
            if kind == "ZONE_START":
                starts[sk] = t
            elif kind == "ZONE_END" and sk in starts:
                data[key][zone].append(((cx, cy), starts.pop(sk), t))

runs_by_dev = collections.defaultdict(list)
for dev, run in kernel:
    runs_by_dev[dev].append(int(run))
keep = {(d, str(r)) for d, rs in runs_by_dev.items() for r in sorted(rs)[-calls:]}

agg = collections.defaultdict(list)  # metric -> values over (dev, run)
for key in sorted(keep):
    t0, t1 = kernel[key]
    us = lambda c: (c - t0) / MHZ
    agg["program (first kernel start -> last kernel end)"].append(us(t1))
    for zone in ZONES:
        spans = data[key].get(zone)
        if not spans:
            continue
        agg[f"{zone} start"].append(min(us(s) for _, s, _ in spans))
        agg[f"{zone} end (last core)"].append(max(us(e) for _, _, e in spans))
        agg[f"{zone} duration (median core)"].append(statistics.median((e - s) / MHZ for _, s, e in spans))

print(f"{len(keep)} (device, call) samples")
print(f"{'metric':52s} {'min':>8s} {'median':>8s} {'max':>8s}")
for name, vals in agg.items():
    print(f"{name:52s} {min(vals):8.1f} {statistics.median(vals):8.1f} {max(vals):8.1f}")
