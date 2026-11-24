# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the last q_stem call per chip from generated/profiler/.logs/profile_log_device.csv (zone_probe.py).
Times in us from the earliest kernel start on that chip. Host-side script, no device."""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    chip = r["PCIe slot"]
    last[chip] = max(last.get(chip, 0), int(r["run host ID"]))
for chip in sorted(last):
    ev = defaultdict(dict)  # (x,y) -> {(risc, zone, type): cycles}
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        key = (r["RISC processor type"], r["zone name"], r["type"])
        c = (int(r["core_x"]), int(r["core_y"]))
        t = int(r["time[cycles since reset]"])
        d = ev[c]
        if r["type"] == "ZONE_START":
            d[key] = min(d.get(key, t), t)
        else:
            d[key] = max(d.get(key, t), t)
    t0 = min(v for d in ev.values() for k, v in d.items() if k[1].endswith("KERNEL") and k[2] == "ZONE_START")
    us = lambda c: (c - t0) / MHZ
    s1, other = [], []
    for c, d in ev.items():
        br = d.get(("BRISC", "BRISC-KERNEL", "ZONE_END"))
        nc = d.get(("NCRISC", "NCRISC-KERNEL", "ZONE_END"))
        s1e = d.get(("TRISC_1", "S1", "ZONE_END"))
        s3s = d.get(("TRISC_1", "S3", "ZONE_START"))
        tr = d.get(("TRISC_1", "TRISC-KERNEL", "ZONE_END"))
        if s3s is None:
            continue
        s1s = d.get(("TRISC_1", "S1", "ZONE_START"))
        rec = (
            c,
            us(s1e) if s1e else None,
            us(tr),
            us(br) if br else None,
            us(nc) if nc else None,
            us(s1s) if s1s else None,
        )
        (s1 if s1e else other).append(rec)

    def stat(name, recs, i):
        v = sorted(r[i] for r in recs if r[i] is not None)
        if v:
            print(f"  {name:28s} n={len(v):3d} min {v[0]:7.1f} med {v[len(v)//2]:7.1f} max {v[-1]:7.1f}")

    allend = max(us(v) for d in ev.values() for k, v in d.items() if k[1].endswith("KERNEL") and k[2] == "ZONE_END")
    print(f"chip {chip}: program end {allend:.1f} us")
    stat("stage-1 cores S1 start", s1, 5)
    stat("stage-1 cores S1 end", s1, 1)
    stat("stage-1 cores compute end", s1, 2)
    stat("stage-1 cores writer end", s1, 3)
    stat("free cores compute end", other, 2)
    stat("free cores writer end", other, 3)
    stat("free cores reader end", other, 4)
