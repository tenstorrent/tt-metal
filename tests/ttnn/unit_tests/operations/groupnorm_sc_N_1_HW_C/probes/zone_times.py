"""Per-core zone durations (us) for the k-th GenericOp of a --profile report dir.
usage: python3 zone_times.py <report_dir> <op_index> <zone_name_prefix> [max_cores]"""
import csv, os, sys
from collections import defaultdict

rep, k, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
maxc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
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
starts, tot = {}, defaultdict(float)
for line in lines[1:]:
    f = [x.strip() for x in line.split(",")]
    if len(f) < len(hdr) - 1 or f[ix["run host ID"]] != run_id:
        continue
    z = f[ix["zone name"]]
    if not (z.startswith(prefix) or z.endswith("-KERNEL")):
        continue
    key = ((int(f[ix["core_x"]]), int(f[ix["core_y"]])), f[ix["RISC processor type"]], z)
    t = int(f[ix["time[cycles since reset]"]])
    if f[ix["type"]] == "ZONE_START":
        starts[key] = t
    elif f[ix["type"]] == "ZONE_END" and key in starts:
        tot[key] += (t - starts.pop(key)) / 1350.0
cores = sorted({k[0] for k in tot})[:maxc]
for c in cores:
    print(
        " ",
        c,
        {
            f"{r}:{z}": f"{d:.2f}"
            for (cc, r, z), d in sorted(tot.items())
            if cc == c and (z.startswith(prefix) or r == "NCRISC")
        },
    )
