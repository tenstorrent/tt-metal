"""Summarize device zones from profile_log_device.csv: per (RISC, zone) — cores, execs, sum/max/p50 duration
per core (ns), and the latest end relative to the earliest kernel start. Usage: python zone_summary.py [csv]."""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    header = f.readline()
    mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]
idx = {c: i for i, c in enumerate(cols)}
starts = {}
per_core = defaultdict(lambda: defaultdict(float))  # (risc, zone) -> core -> cycles
execs = defaultdict(lambda: defaultdict(int))
ends = defaultdict(list)
t0 = None
for r in rows[1:]:
    core = (r[idx["core_x"]].strip(), r[idx["core_y"]].strip())
    risc = r[idx["RISC processor type"]].strip()
    zone = r[idx["zone name"]].strip()
    typ = r[idx["type"]].strip()
    t = int(r[idx["time[cycles since reset]"]])
    key = (core, risc, zone)
    if typ == "ZONE_START":
        starts.setdefault(key, []).append(t)
        if zone.endswith("-KERNEL"):
            t0 = t if t0 is None else min(t0, t)
    elif typ == "ZONE_END" and starts.get(key):
        s = starts[key].pop()
        per_core[(risc, zone)][core] += t - s
        execs[(risc, zone)][core] += 1
        ends[(risc, zone)].append(t)
ns = 1000.0 / mhz
print(
    f"{'risc':7} {'zone':26} {'cores':>5} {'exec':>5} {'max_ns':>10} {'p50_ns':>10} {'min_ns':>10} {'last_end_ns':>12}"
)
for (risc, zone), d in sorted(per_core.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    v = list(d.values())
    e = max(execs[(risc, zone)].values())
    last = (max(ends[(risc, zone)]) - t0) * ns if t0 is not None else 0
    print(
        f"{risc:7} {zone:26} {len(v):5d} {e:5d} {max(v)*ns:10.0f} {statistics.median(v)*ns:10.0f} {min(v)*ns:10.0f} {last:12.0f}"
    )
