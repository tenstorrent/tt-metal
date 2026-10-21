"""Sum the GLAZY-* counters (UNPACK thread) of the global SDPA op over all cores and devices."""
import collections
import csv
import glob
import sys

R = sys.argv[1]
rows = list(csv.DictReader(open(glob.glob(R + "/ops_perf_results_*.csv")[0])))
i0 = [i for i, r in enumerate(rows) if r["OP CODE"].endswith("-start")][-1]
win = {}
for r in rows[i0 + 1 :]:
    if r["OP CODE"] == "RingJointSDPADeviceOperation" and r["DEVICE ID"] not in win:
        win[r["DEVICE ID"]] = (int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"]))
tot = collections.Counter()
with open(glob.glob(R + "/profile_log_device.csv")[0]) as fh:
    next(fh)
    next(fh)
    for line in fh:
        p = line.split(",", 14)
        if not p[10].startswith("GLAZY-") or p[3] != "TRISC_0" or p[0] not in win:
            continue
        a, b = win[p[0]]
        t = int(p[5])
        if a <= t <= b + 200000:
            tot[p[10]] += int(p[6])
g = tot["GLAZY-GROUPS"]
print(f"devices {len(win)}  steps {tot['GLAZY-STEPS']}  row groups rescaled {g}")
for k, label in (
    ("GLAZY-ONE", "max unchanged (exact skip possible)"),
    ("GLAZY-T4", "all factors >= 2^-4"),
    ("GLAZY-T8", "all factors >= 2^-8"),
):
    print(f"  {label:40s} {tot[k]:10d}  = {100.0 * tot[k] / max(g, 1):5.1f}% of row groups")
