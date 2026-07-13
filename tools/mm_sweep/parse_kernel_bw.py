import csv, sys

csv_path, total_bytes = sys.argv[1], int(sys.argv[2])
risc_filter = sys.argv[3] if len(sys.argv) > 3 else None  # e.g. "BRISC" to isolate the reader
FREQ = 1.35e9
# collect BRISC-KERNEL start/end per (core,run)
starts = {}
durations = []  # list of dicts per run
# We track per core a queue of start times; pair with ends in order
from collections import defaultdict

ev = defaultdict(list)  # (x,y)-> list of (type,cycle)
with open(csv_path) as f:
    r = csv.reader(f)
    next(r)
    next(r)
    for row in r:
        if len(row) < 12:
            continue
        x, y, risc, zone, typ, cyc = row[1], row[2], row[3], row[10].strip(), row[11].strip(), row[5]
        if zone.endswith("-KERNEL") and (risc_filter is None or risc == risc_filter):
            ev[(x, y, risc)].append((typ, int(cyc)))
# pair starts/ends per core -> list of durations (one per run)
per_core_runs = {}
for core, lst in ev.items():
    ds = []
    st = None
    for typ, cyc in lst:
        if typ == "ZONE_START":
            st = cyc
        elif typ == "ZONE_END" and st is not None:
            ds.append(cyc - st)
            st = None
    per_core_runs[core] = ds
nruns = min(len(v) for v in per_core_runs.values())
# per run: max across cores
run_max = []
for i in range(nruns):
    run_max.append(max(v[i] for v in per_core_runs.values()))
# skip first (cold), report min & median of rest
steady = sorted(run_max[1:]) if nruns > 1 else run_max
best = steady[0]
med = steady[len(steady) // 2]


def bw(cyc):
    return total_bytes / (cyc / FREQ) / 1e9


print(
    f"ncores={len(per_core_runs)} nruns={nruns} best_cyc={best} med_cyc={med} "
    f"BW_best={bw(best):.1f} BW_med={bw(med):.1f} GB/s"
)
