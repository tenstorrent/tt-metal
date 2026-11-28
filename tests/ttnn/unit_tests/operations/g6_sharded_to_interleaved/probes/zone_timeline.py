"""Per-core zone timeline from profile_log_device.csv for the runs of one ops-CSV group.

Usage: python zone_timeline.py <ops csv> <group index> [calls per group=12] [drop=2]
Prints, per group (medians over its runs), the spread across cores of: kernel start (rel. to the earliest core),
issue time (sum of S2I-STORE-BLOCK), barrier time, kernel end (rel.), and which cores end last.
"""
import csv
import statistics
import sys
from collections import defaultdict

ops_csv, gidx = sys.argv[1], int(sys.argv[2])
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 12
drop = int(sys.argv[4]) if len(sys.argv) > 4 else 2
import os

# The report dir keeps its own copy of the device log; .logs/ holds only the latest run's.
log = os.path.join(os.path.dirname(ops_csv), "profile_log_device.csv")
if not os.path.exists(log):
    log = ops_csv.rsplit("/reports/", 1)[0] + "/.logs/profile_log_device.csv"
rows = [r for r in csv.DictReader(open(ops_csv)) if r.get("OP CODE", "GenericOp").startswith("GenericOp")]
run_ids = {r["GLOBAL CALL COUNT"] for r in rows[gidx * calls + drop : (gidx + 1) * calls]}

ev = defaultdict(lambda: defaultdict(list))  # run -> (x,y,risc) -> [(name, type, t)]
with open(log) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        if r["run host ID"] in run_ids:
            ev[r["run host ID"]][(int(r["core_x"]), int(r["core_y"]), r["RISC processor type"])].append(
                (r["zone name"], r["type"], int(r["time[cycles since reset]"]))
            )

stats = defaultdict(list)
last_cores = defaultdict(int)
for run, cores in ev.items():
    per = {}
    for core, events in cores.items():
        d = defaultdict(int)
        opened = {}
        for name, typ, t in events:
            if typ == "ZONE_START":
                opened[name] = t
                if name.endswith("-KERNEL"):
                    d["kstart"] = t
            elif typ == "ZONE_END" and name in opened:
                d[name] += t - opened.pop(name)
                if name.endswith("-KERNEL"):
                    d["kend"] = t
        if "kstart" in d:
            per[core] = d
    t0 = min(d["kstart"] for d in per.values())
    for core, d in per.items():
        stats["start"].append(d["kstart"] - t0)
        stats["issue"].append(d["S2I_STORE_BLOCK"])
        stats["barrier"].append(d["S2I_DRAIN"])
        stats["end"].append(d["kend"] - t0)
    for core, d in sorted(per.items(), key=lambda kv: -kv[1]["kend"])[:3]:
        last_cores[core] += 1


def q(v):
    v = sorted(v)
    return f"min={v[0]:5d} p50={v[len(v)//2]:5d} p90={v[int(len(v)*0.9)]:5d} max={v[-1]:5d}"


print(f"group {gidx}: {len(ev)} runs, cycles @1350MHz")
for k in ("start", "issue", "barrier", "end"):
    print(f"  {k:8s} {q(stats[k])}")
print("  last-ending cores (x,y,risc: count):", sorted(last_cores.items(), key=lambda kv: -kv[1])[:8])
