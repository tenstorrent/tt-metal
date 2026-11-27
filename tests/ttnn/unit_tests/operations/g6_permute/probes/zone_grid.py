"""Print one zone's per-core duration (us) as a grid of logical-ish cores (virtual x, y). Usage: zone_grid.py RISC ZONE [csv]."""
import csv
import sys
from collections import defaultdict

risc_q, zone_q = sys.argv[1], sys.argv[2]
path = sys.argv[3] if len(sys.argv) > 3 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    header = f.readline()
    mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
cols = [c.strip() for c in rows[0]]
idx = {c: i for i, c in enumerate(cols)}
starts, dur = {}, defaultdict(float)
for r in rows[1:]:
    if r[idx["RISC processor type"]].strip() != risc_q or r[idx["zone name"]].strip() != zone_q:
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]].strip() == "ZONE_START":
        starts[core] = t
    else:
        dur[core] += t - starts.pop(core)
xs = sorted({c[0] for c in dur})
ys = sorted({c[1] for c in dur})
print("y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:3d} " + " ".join(f"{dur[(x, y)] / mhz:5.0f}" if (x, y) in dur else "    -" for x in xs))
