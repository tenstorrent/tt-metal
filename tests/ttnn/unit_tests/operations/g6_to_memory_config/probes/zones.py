"""Summarize device zones of the last profiled run: per zone, per RISC -> cores, max/p50 of per-core sums (us)."""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
starts = {}
sums = defaultdict(lambda: defaultdict(float))  # (run, zone, risc) -> core -> cycles
kspan = defaultdict(dict)
for r in rows[1:]:
    run = r[ix["run host ID"]].strip()
    core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip())
    risc = r[ix["RISC processor type"]].strip()
    zone = r[ix["zone name"]].strip()
    typ = r[ix["type"]].strip()
    t = int(r[ix["time[cycles since reset]"]])
    key = (run, core, risc, zone)
    if typ == "ZONE_START":
        starts[key] = t
    elif typ == "ZONE_END" and key in starts:
        sums[(run, zone, risc)][core] += t - starts.pop(key)
for (run, zone, risc), per_core in sorted(sums.items()):
    if zone.endswith("-FW"):
        continue
    vals = sorted(per_core.values())
    print(
        f"run {run:>6} {risc:6} {zone:26} cores {len(vals):4} max {vals[-1]/freq:9.2f} p50 {statistics.median(vals)/freq:9.2f} us"
    )
