import csv, glob, os, sys

cands = glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv")
if not cands:
    print("NOCSV")
    sys.exit(0)
newest = max(cands, key=os.path.getmtime)
with open(newest) as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("EMPTY")
    sys.exit(0)
dc = [h for h in rows[0] if "DEVICE KERNEL DURATION" in h]
if not dc:
    print("NODURCOL")
    sys.exit(0)
conv = [r for r in rows if r.get("OP CODE", "").strip() == "Conv2dDeviceOperation"]
if not conv:
    print("NOCONV")
    sys.exit(0)
warm = conv[1] if len(conv) >= 2 else conv[-1]
print(warm[dc[0]].strip())
