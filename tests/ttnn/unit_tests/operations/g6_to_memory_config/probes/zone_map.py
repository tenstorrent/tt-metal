"""Per-core map of one zone (sum per core, us) for one run: zone_map.py <zone> [run_id] [log]."""
import csv
import sys
from collections import defaultdict

zone_name = sys.argv[1]
run_id = sys.argv[2] if len(sys.argv) > 2 else None
path = sys.argv[3] if len(sys.argv) > 3 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
ix = {h.strip(): i for i, h in enumerate(rows[0])}
starts, acc = {}, defaultdict(float)
runs = []
for r in rows[1:]:
    run = r[ix["run host ID"]].strip()
    if run not in runs:
        runs.append(run)
    if r[ix["zone name"]].strip() != zone_name:
        continue
    if run_id is not None and run != run_id:
        continue
    core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
    risc = r[ix["RISC processor type"]].strip()
    key = (core, risc)
    t = int(r[ix["time[cycles since reset]"]])
    if r[ix["type"]].strip() == "ZONE_START":
        starts[key] = t
    elif key in starts:
        acc[core] += (t - starts.pop(key)) / freq
xs = sorted({c[0] for c in acc})
ys = sorted({c[1] for c in acc})
print("runs:", runs, "zone", zone_name, "run", run_id)
print("y\\x " + "".join(f"{x:6}" for x in xs))
for y in ys:
    print(f"{y:3} " + "".join(f"{acc.get((x, y), 0):6.0f}" for x in xs))
