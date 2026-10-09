# Summarize sep.py processes: per process the median device time over its calls; per round r and case, effect = mean of
# the two "off" processes minus mean of the two "on" processes; A/A = on1 - on2 and off1 - off2 of the round. Verdict rule
# (as in the alternations, written before the run): "separates" when every round's effect has the same sign and the
# median effect is larger than every A/A magnitude; "equal" otherwise. usage: sep_sum.py <dir> <on1,on2,off1,off2>
import csv, glob, os, re, statistics, sys
from collections import defaultdict
on1, on2, off1, off2 = sys.argv[2].split(",")
val = defaultdict(dict)
for rd in sorted(glob.glob(sys.argv[1] + "/out_*")):
    m = re.match(r"out_(\d+)_(\w+)_(\w+)$", os.path.basename(rd))
    if not m:
        continue
    r, case, form = int(m.group(1)), m.group(2), m.group(3)
    fs = glob.glob(f"{rd}/reports/*/ops_perf_results_*.csv")
    if not fs:
        print("no report", rd); continue
    per = defaultdict(float); cur = None
    for row in csv.DictReader(open(fs[0])):
        if row.get("OP TYPE") == "signpost":
            p = row["OP CODE"].split("|"); cur = p[2] if len(p) == 3 else None; continue
        if cur is None:
            continue
        try:
            per[cur] += float(row["DEVICE KERNEL DURATION [ns]"])
        except ValueError:
            pass
    if per:
        val[(case, r)][form] = statistics.median(per.values())
        print(f"round {r} {case} {form}: {val[(case, r)][form]:.0f} ns over {len(per)} calls")
for case in sorted({k[0] for k in val}):
    rounds = sorted(r for (c, r) in val if c == case and all(f in val[(c, r)] for f in (on1, on2, off1, off2)))
    eff = [(val[(case, r)][off1] + val[(case, r)][off2]) / 2 - (val[(case, r)][on1] + val[(case, r)][on2]) / 2 for r in rounds]
    aa = [val[(case, r)][on1] - val[(case, r)][on2] for r in rounds] + [val[(case, r)][off1] - val[(case, r)][off2] for r in rounds]
    base = statistics.median([val[(case, r)][f] for r in rounds for f in (on1, on2)])
    e = statistics.median(eff); big = max(abs(x) for x in aa)
    same = all(x > 0 for x in eff) or all(x < 0 for x in eff)
    v = ("separates, the edit faster" if e > 0 else "separates, the edit slower") if same and abs(e) > big else "equal"
    print(f"| {case} | with {base:.0f} | without minus with {e:+.0f} ({', '.join(f'{x:+.0f}' for x in eff)}) {100 * e / base:+.2f} percent | A/A {', '.join(f'{x:+.0f}' for x in aa)} | {v} |")
