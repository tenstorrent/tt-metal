"""Per signposted case of one tracy run: per op code, the median DEVICE KERNEL DURATION over the measured calls, and the
perf counter metric columns of the SoftmaxDeviceOperation rows when present.
usage: sum_sm.py <tracy out dir> <variant> <round> [--counters]"""
import csv, glob, statistics, collections, sys
d, v, r = sys.argv[1], sys.argv[2], sys.argv[3]
fs = glob.glob(d + "/reports/*/ops_perf_results_*.csv") + glob.glob(d + "/ops_perf_results_*.csv")
if not fs:
    print("SMR NOREPORT", v, r); sys.exit(0)
rows = list(csv.DictReader(open(fs[0])))
case = None; agg = collections.defaultdict(list); ctr = collections.defaultdict(lambda: collections.defaultdict(list))
ccols = [c for c in (rows[0].keys() if rows else []) if any(k in c for k in ("Median", "Avg")) and "DURATION" not in c]
for row in rows:
    if row.get("OP TYPE") == "signpost":
        case = row.get("OP CODE"); continue
    if case is None: continue
    try: val = float(row["DEVICE KERNEL DURATION [ns]"])
    except (KeyError, ValueError): continue
    op = row["OP CODE"].replace(" ", "_")
    agg[(case, op)].append(val)
    if "--counters" in sys.argv and "Softmax" in op:
        for c in ccols:
            try: ctr[(case, op)][c].append(float(row[c]))
            except (KeyError, ValueError): pass
for (case, op), vals in sorted(agg.items()):
    print("SMR", case, v, r, op, len(vals), round(statistics.median(vals)), " ".join(str(round(x)) for x in vals))
for (case, op), cols in sorted(ctr.items()):
    for c, vals in cols.items():
        if vals:
            print("SMC", case, v, op, c.replace(" ", "_"), round(statistics.median(vals), 3))
