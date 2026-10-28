"""Total DEVICE KERNEL DURATION [ns] between BEGIN/END signposts of an ops_perf_results csv.

Prints, per tag, the mean over repeats of the summed kernel time and the number of programs per call.
Usage: sum_signposts.py <ops_perf_results.csv>
"""
import collections
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
totals = collections.defaultdict(list)
programs = collections.defaultdict(list)
ops = collections.defaultdict(collections.Counter)
tag = None
for r in rows:
    code = r["OP CODE"]
    if r.get("OP TYPE") == "signpost":
        if code.startswith("BEGIN "):
            tag, total, count = code[6:], 0.0, 0
        elif code.startswith("END ") and tag:
            totals[tag].append(total)
            programs[tag].append(count)
            tag = None
        continue
    if tag is None:
        continue
    duration = r.get("DEVICE KERNEL DURATION [ns]", "")
    if duration:
        total += float(duration)
        count += 1
        ops[tag][code] += 1

for t in totals:
    reps = len(totals[t])
    breakdown = ", ".join(f"{k} x{v // reps}" for k, v in ops[t].items())
    print(f"{t:36s} {statistics.mean(totals[t]):10.0f} ns  programs={programs[t][0]:2d}  [{breakdown}]")
