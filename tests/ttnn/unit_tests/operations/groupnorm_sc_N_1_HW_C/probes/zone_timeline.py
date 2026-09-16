"""Zone timeline (us) of one core for the k-th GenericOp of a --profile report dir.
usage: python3 zone_timeline.py <report_dir> <op_index> [core_x core_y] [risc_filter]
Prints per RISC every zone occurrence: start offset from the earliest KERNEL start on that core, and duration."""
import csv, os, sys
from collections import defaultdict

rep, k = sys.argv[1], int(sys.argv[2])
core = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
risc_filter = sys.argv[5] if len(sys.argv) > 5 else None
ops = [
    r
    for r in csv.DictReader(open(f"{rep}/" + [f for f in os.listdir(rep) if f.startswith("ops_perf")][0]))
    if r["OP CODE"] == "GenericOpDeviceOperation"
]
run_id = ops[k]["GLOBAL CALL COUNT"]
print(f"op {k}: {float(ops[k]['DEVICE KERNEL DURATION [ns]'])/1000:.2f} us, cores {ops[k]['CORE COUNT']}")
lines = open(f"{rep}/profile_log_device.csv").read().splitlines()[1:]
hdr = [h.strip() for h in lines[0].split(",")]
ix = {h: i for i, h in enumerate(hdr)}
starts, events = {}, defaultdict(list)
for line in lines[1:]:
    f = [x.strip() for x in line.split(",")]
    if len(f) < len(hdr) - 1 or f[ix["run host ID"]] != run_id:
        continue
    c = (int(f[ix["core_x"]]), int(f[ix["core_y"]]))
    r = f[ix["RISC processor type"]]
    z = f[ix["zone name"]]
    t = int(f[ix["time[cycles since reset]"]])
    key = (c, r, z)
    if f[ix["type"]] == "ZONE_START":
        starts[key] = t
    elif f[ix["type"]] == "ZONE_END" and key in starts:
        events[c].append((r, z, starts.pop(key), t))
cores = sorted(events)
if core is None:
    core = cores[0]
ev = events[core]
t0 = min(s for (r, z, s, e) in ev if z.endswith("-KERNEL"))
print("core", core, "(of", len(cores), "cores)")
for r in sorted({e[0] for e in ev}):
    if risc_filter and r != risc_filter:
        continue
    print(" ", r)
    for rr, z, s, e in sorted([e for e in ev if e[0] == r], key=lambda e: e[2]):
        print(f"    {z:28s} start {(s-t0)/1350:7.2f}  dur {(e-s)/1350:7.2f}")
