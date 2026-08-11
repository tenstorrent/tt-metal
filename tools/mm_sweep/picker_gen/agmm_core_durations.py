#!/usr/bin/env python3
"""Per-core kernel durations for the LAST op in the profiler CSV, split matmul-core vs everything else.

The whole-op median cannot say WHO is slow. The makespan is the max over cores, and the AGMM runs two very
different kinds of core: the ~80 matmul cores (which have TRISCs) and the fabric mux / dispatch cores (data
movement only). If an ablation's cost shows up only on the mux cores it is fabric drain; if every matmul core
slows down together it is contention for a shared resource.

usage: agmm_core_durations.py [profile_log_device.csv]
"""
import csv
import statistics
import sys

FREQ = 1.35e9
CSV = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"

rows = list(csv.reader(open(CSV)))
hdr = next(i for i, r in enumerate(rows) if "zone name" in [c.strip().lower() for c in r])
idx = {c.strip().lower(): i for i, c in enumerate(rows[hdr])}
tcol = next(c for c in idx if c.startswith("time[cycles"))

# LAST zone pair per (device, core, risc): the CSV holds every op in the process.
dur = {}
for r in rows[hdr + 1 :]:
    if len(r) <= max(idx.values()) or not r[idx["zone name"]].strip().endswith("-KERNEL"):
        continue
    key = (
        r[idx["pcie slot"]].strip(),
        r[idx["core_x"]].strip(),
        r[idx["core_y"]].strip(),
        r[idx["risc processor type"]].strip(),
    )
    e = dur.setdefault(key, {})
    if r[idx["type"]].strip() == "ZONE_START":
        e["s"] = int(r[idx[tcol]])
    elif "s" in e:
        e["d"] = (int(r[idx[tcol]]) - e["s"]) / FREQ * 1e6

# A core that ran a TRISC is a matmul core; the rest are mux / dispatch.
tri = {(d, x, y) for (d, x, y, ri) in dur if ri.startswith("TRISC")}
mm = sorted(v["d"] for (d, x, y, ri), v in dur.items() if "d" in v and (d, x, y) in tri)
other = sorted((v["d"], d, x, y, ri) for (d, x, y, ri), v in dur.items() if "d" in v and (d, x, y) not in tri)
print(f"  matmul-core riscs n={len(mm):4d} min {mm[0]:6.1f} med {statistics.median(mm):6.1f} max {mm[-1]:6.1f}")
if other:
    med = statistics.median([o[0] for o in other])
    print(
        f"  non-matmul cores n={len(other):4d} min {other[0][0]:6.1f} med {med:6.1f} max {other[-1][0]:6.1f}"
        f"  slowest={other[-1][1:]}"
    )
