# Per signposted case in one tracy report: median device kernel duration (ns) of the 6 measured typecast calls.
import csv, glob, statistics, sys
d, v, r = sys.argv[1:4]
fs = glob.glob(d + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("RAB NOREPORT", v, r); sys.exit(0)
cur, vals = None, {}
for row in csv.DictReader(open(fs[0])):
    if (row.get("OP TYPE") or "").lower() == "signpost":
        cur = row.get("OP CODE"); vals[cur] = []; continue
    if cur and "ypecast" in (row.get("OP CODE") or ""):
        try: vals[cur].append(float(row["DEVICE KERNEL DURATION [ns]"]))
        except (KeyError, ValueError): pass
for c, xs in vals.items():
    last = xs[-6:]
    print("RAB", c, v, r, len(xs), round(statistics.median(last)) if last else -1, " ".join(str(int(x)) for x in last))
