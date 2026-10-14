# Parse one tracy output dir of prof_rr2.py: per case (signpost segments of the ops report), the embedding_bw device
# kernel durations; from the device log, the RR_* timestamped sums and the perf counter records of those calls.
import csv, glob, json, statistics, sys
from collections import defaultdict

o, tag = sys.argv[1], sys.argv[2]
fs = glob.glob(o + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print("SFOP", tag, "NOREPORT"); sys.exit(0)
rows = list(csv.DictReader(open(fs[0])))
case, call2case, per = None, {}, defaultdict(list)
for r in rows:
    if r.get("OP TYPE") == "signpost":
        case = r["OP CODE"]; continue
    if case and r.get("DEVICE KERNEL DURATION [ns]") and "Embedding" in r.get("OP CODE", ""):
        per[case].append(float(r["DEVICE KERNEL DURATION [ns]"]))
        call2case[str(r.get("GLOBAL CALL COUNT"))] = case
for c, xs in per.items():
    print("SFOP", tag, c, len(xs), statistics.median(xs), " ".join("%.0f" % x for x in xs))
logs = glob.glob(o + "/**/profile_log_device.csv", recursive=True)
if not logs:
    sys.exit(0)
ts = defaultdict(lambda: defaultdict(float))
pc = defaultdict(lambda: defaultdict(float))
with open(logs[0]) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        c = call2case.get(str(r.get("run host ID")))
        if not c:
            continue
        name = r.get("zone name", "")
        if name.startswith("RR_"):
            ts[c][(name, r["RISC processor type"])] += float(r["data"])
        elif r.get("timer_id") == "9090" and r.get("meta data"):
            try:
                md = json.loads(r["meta data"].replace(";", ",").replace("'", '"'))
            except Exception:
                continue
            pc[c][(str(md.get("counter type")), r["RISC processor type"])] += float(md.get("value", md.get("counter value", 0)) or 0)
for c in ts:
    print("SFTS", tag, c, " ".join("%s/%s=%.0f" % (k[0], k[1], v) for k, v in sorted(ts[c].items())))
for c in pc:
    print("SFPC", tag, c, " ".join("%s/%s=%.0f" % (k[0], k[1], v) for k, v in sorted(pc[c].items())))
