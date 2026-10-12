# Device kernel durations of the dropout op in one tracy report: count, median of the last 6, the last 6.
import csv, glob, statistics, sys
d, case, v, r = sys.argv[1:5]
fs = glob.glob(d + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("RAB NOREPORT", case, v, r); sys.exit(0)
vals = []
for row in csv.DictReader(open(fs[0])):
    if "ropout" not in (row.get("OP CODE") or ""): continue
    try: vals.append(float(row["DEVICE KERNEL DURATION [ns]"]))
    except (KeyError, ValueError): pass
last = vals[-6:]
print("RAB", case, v, r, len(vals), round(statistics.median(last)) if last else -1, " ".join(str(int(x)) for x in last))
