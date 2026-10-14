# Per case (signpost segments of the ops report): device kernel duration and per-RISC kernel durations of each op code.
import csv, glob, statistics, sys
from collections import defaultdict

o, tag = sys.argv[1], sys.argv[2]
fs = glob.glob(o + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("SFOP", tag, "NOREPORT"); sys.exit(0)
case, per = None, defaultdict(lambda: defaultdict(list))
cols = ["DEVICE KERNEL DURATION [ns]"] + [f"DEVICE {r} KERNEL DURATION [ns]" for r in ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")]
for r in csv.DictReader(open(fs[0])):
    if r.get("OP TYPE") == "signpost":
        case = r["OP CODE"]; continue
    if case and r.get(cols[0]):
        for c in cols:
            try:
                per[(case, r["OP CODE"].replace(" ", "_"))][c].append(float(r.get(c) or "nan"))
            except ValueError:
                pass
for (c, op), d in per.items():
    xs = d[cols[0]]
    risc = " ".join("%s=%.0f" % (k.split()[1], statistics.median(d[k])) for k in cols[1:] if d[k])
    print("SFOP", tag, c, op, len(xs), statistics.median(xs), " ".join("%.0f" % x for x in xs), "|", risc)
