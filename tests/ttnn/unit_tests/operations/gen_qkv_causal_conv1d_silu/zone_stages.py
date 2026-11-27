# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage table of gen_qkv_causal_conv1d_silu from its permanent MaybeDeviceZoneScope zones (host-side, no device).

For each program run (run host ID) in profile_log_device.csv, or only the one given by --run:
  - per RISC: *-KERNEL span max / p50 across cores (core balance) and the max user-marker count (a RISC near
    250 optional markers has silently lost zones);
  - per (zone, RISC): cores, summed duration per core (p50 / max), executions per core, last end (p50 / max).

    python3 tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/zone_stages.py [csv] [--run N] [--mhz 1350]
"""
import csv
import sys
from collections import defaultdict

args = sys.argv[1:]
path = "generated/profiler/.logs/profile_log_device.csv"
only_run = None
mhz = None
i = 0
while i < len(args):
    if args[i] == "--run":
        only_run = int(args[i + 1])
        i += 2
    elif args[i] == "--mhz":
        mhz = float(args[i + 1])
        i += 2
    else:
        path = args[i]
        i += 1

f = open(path)
header = f.readline()
if mhz is None:
    mhz = 1350.0
    if "CHIP_FREQ[MHz]:" in header:
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip())
rows = list(csv.DictReader(f, skipinitialspace=True))
runs = sorted({int(r["run host ID"]) for r in rows})
if only_run is not None:
    runs = [only_run]


def q(v):
    v = sorted(v)
    return v[len(v) // 2], v[-1]


for run in runs:
    events = defaultdict(list)  # (core, risc, zone) -> list of (type, t)
    markers = defaultdict(int)
    for r in rows:
        if int(r["run host ID"]) != run:
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        key = (c, r["RISC processor type"], r["zone name"])
        events[key].append((r["type"], int(r["time[cycles since reset]"])))
        if not r["zone name"].endswith("KERNEL") and not r["zone name"].endswith("FW"):
            markers[(c, r["RISC processor type"])] += 1
    spans = {}
    total = defaultdict(int)
    count = defaultdict(int)
    last_end = {}
    first_start = {}
    for key, ev in events.items():
        ev.sort(key=lambda x: x[1])
        stack = []
        for typ, t in ev:
            if typ == "ZONE_START":
                stack.append(t)
            elif typ == "ZONE_END" and stack:
                s = stack.pop()
                total[key] += t - s
                count[key] += 1
                last_end[key] = max(last_end.get(key, 0), t)
                first_start[key] = min(first_start.get(key, t), s) if key in first_start else s
    kernel_keys = [k for k in total if k[2].endswith("KERNEL")]
    if not kernel_keys:
        continue
    t0 = min(first_start[k] for k in kernel_keys)
    t_end = max(last_end[k] for k in kernel_keys)
    print(
        f"=== run {run}: program span {(t_end - t0) / mhz:.2f} us @ {mhz:.0f} MHz; "
        f"max user markers per RISC {max(markers.values(), default=0)}"
    )
    by_risc = defaultdict(list)
    for k in kernel_keys:
        by_risc[k[1]].append(total[k] / mhz)
    for risc, v in sorted(by_risc.items()):
        p50, mx = q(v)
        print(
            f"  {risc:8s} KERNEL span  cores={len(v):3d}  p50 {p50:8.2f}  max {mx:8.2f}  max/p50 {mx / p50 if p50 else 0:.2f}"
        )
    agg = defaultdict(list)
    for k in total:
        if k[2].endswith("KERNEL"):
            continue
        agg[(k[2], k[1])].append((total[k] / mhz, count[k], (last_end[k] - t0) / mhz))
    for (z, risc), v in sorted(agg.items(), key=lambda kv: (kv[0][1], -q([x[0] for x in kv[1]])[0])):
        d = q([x[0] for x in v])
        n = q([x[1] for x in v])
        e = q([x[2] for x in v])
        print(
            f"    {risc:8s} {z:24s} cores={len(v):3d} sum p50 {d[0]:8.2f} max {d[1]:8.2f}  "
            f"exec p50 {n[0]:3d}  last_end p50 {e[0]:8.2f} max {e[1]:8.2f}"
        )
