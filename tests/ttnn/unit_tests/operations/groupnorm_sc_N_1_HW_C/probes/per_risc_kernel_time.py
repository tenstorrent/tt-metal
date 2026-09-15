"""Per-RISC kernel durations (us) for the k-th GenericOp of a --profile report dir.

usage: python3 per_risc_kernel_time.py <report_dir> <op_index (0-based, GenericOp rows in order)> [core_x core_y]
Prints, per core (or the given core) and per RISC, end - start of the *-KERNEL zone in microseconds.
"""
import csv
import sys
from collections import defaultdict

rep, k = sys.argv[1], int(sys.argv[2])
core = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
ops = [
    r
    for r in csv.DictReader(open(f"{rep}/" + [f for f in __import__("os").listdir(rep) if f.startswith("ops_perf")][0]))
    if r.get("OP CODE") == "GenericOpDeviceOperation"
]
op = ops[k]
run_id = op["GLOBAL CALL COUNT"] if "GLOBAL CALL COUNT" in op else None
print(
    "op", k, "kernel us", float(op["DEVICE KERNEL DURATION [ns]"]) / 1000, "cores", op["CORE COUNT"], "run id", run_id
)
lines = open(f"{rep}/profile_log_device.csv").read().splitlines()[1:]
hdr = [h.strip() for h in lines[0].split(",")]
ix = {h: i for i, h in enumerate(hdr)}
start, end = {}, {}
for line in lines[1:]:
    f = [x.strip() for x in line.split(",")]
    if len(f) < len(hdr) - 1:
        continue
    if f[ix["run host ID"]] != run_id:
        continue
    zone = f[ix["zone name"]]
    if not zone.endswith("-KERNEL"):
        continue
    key = ((int(f[ix["core_x"]]), int(f[ix["core_y"]])), f[ix["RISC processor type"]])
    t = int(f[ix["time[cycles since reset]"]])
    if f[ix["type"]] == "ZONE_START":
        start[key] = t
    elif f[ix["type"]] == "ZONE_END":
        end[key] = t
per_core = defaultdict(dict)
for key in start:
    if key in end:
        per_core[key[0]][key[1]] = (end[key] - start[key]) / 1350.0
t0 = min(start.values())
for c in sorted(per_core):
    if core and c != core:
        continue
    print(
        c,
        {r: f"{d:.2f}" for r, d in sorted(per_core[c].items())},
        "start offsets:",
        {r: f"{(start[(c, r)] - t0) / 1350.0:.2f}" for r in sorted(per_core[c])},
    )
