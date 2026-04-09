import csv
import sys
from collections import defaultdict

zone_name = sys.argv[2] if len(sys.argv) > 2 else "REDUCE-MATMUL-TILE"
path = sys.argv[1]

starts = defaultdict(list)
ends = defaultdict(list)

with open(path) as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if len(row) >= 12 and zone_name in row[10]:
            risc = row[3]
            core = (row[1], row[2])
            cycles = int(row[5])
            if row[11] == "ZONE_START":
                starts[(risc, core)].append(cycles)
            elif row[11] == "ZONE_END":
                ends[(risc, core)].append(cycles)

for risc in ["TRISC_1"]:
    all_d = []
    first_only = []
    cores = sorted(set(k[1] for k in starts.keys() if k[0] == risc))
    for core in cores:
        s = starts[(risc, core)]
        e = ends[(risc, core)]
        n = min(len(s), len(e))
        d = [e[i] - s[i] for i in range(n)]
        first_only.append(d[0])
        all_d.extend(d[1:])
    if all_d:
        sorted_d = sorted(all_d)
        median = sorted_d[len(sorted_d) // 2]
        print(
            f"{risc}: excl first call: {len(all_d)} calls, median={median}, mean={sum(all_d)/len(all_d):.1f}, min={min(all_d)}, max={max(all_d)}"
        )
