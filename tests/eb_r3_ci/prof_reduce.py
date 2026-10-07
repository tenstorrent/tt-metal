#!/usr/bin/env python3
"""Round 3 eltwise binary: device kernel duration per test node and op from the tracy ops reports of prof_card.sh runs.
usage: prof_reduce.py <prof dir> <run tag> [--md out] [--ops OP1,OP2] [--col "<ops report column>"]
reads out_<tag>_<i>_<variant>/reports/*/ops_perf_results_*.csv. eb_prof_plugin puts a signpost "<nodeid>#<k>" before
repetition k of each test; repetition 0 is the warm-up and is dropped. Per repetition the durations of every launch of an op
code are summed; per run the median over the repetitions; the table compares every variant with main: median of the per-run
medians, the change, and the spread (the larger range of per-run medians of the two sides). 'slower' marks a change above the
spread."""
import csv, glob, os, statistics, sys

KEEP0 = os.environ.get("EB_KEEP_REP0") == "1"  # one repetition per test (EB_REPS=1): keep it
from collections import defaultdict

d, tag = sys.argv[1], sys.argv[2]
md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
COL = sys.argv[sys.argv.index("--col") + 1] if "--col" in sys.argv else "DEVICE KERNEL DURATION [ns]"
only = set(sys.argv[sys.argv.index("--ops") + 1].split(",")) if "--ops" in sys.argv else None
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # node -> op -> variant -> [per-run median]
order = []
runs = [r for t in tag.split(",") for r in sorted(glob.glob(os.path.join(d, f"out_{t}_*_*")), key=lambda p: int(os.path.basename(p).split("_")[2]))]
for rd in runs:
    v = os.path.basename(rd).split("_", 3)[3]
    fs = glob.glob(os.path.join(rd, "reports", "*", "ops_perf_results_*.csv"))
    if not fs:
        print(f"# no report in {rd}", file=sys.stderr)
        continue
    cur = None
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # node -> op -> rep -> sum
    rows = list(csv.DictReader(open(fs[0])))
    if not rows or "OP TYPE" not in rows[0]:
        print(f"# {os.path.basename(rd)}: report without host op data (no tracy host capture), skipped", file=sys.stderr)
        continue
    for row in rows:
        if row["OP TYPE"] == "signpost":
            node, _, k = row["OP CODE"].rpartition("#")
            cur = (node, int(k)) if k.isdigit() else None
            if cur and node not in order:
                order.append(node)
            continue
        if cur is None or (cur[1] == 0 and not KEEP0):
            continue
        try:
            per[cur[0]][row["OP CODE"]][cur[1]] += float(row[COL])
        except ValueError:
            pass
    for node, ops in per.items():
        for op, reps in ops.items():
            data[node][op][v].append(statistics.median(reps.values()))
variants = sorted({v for n in data.values() for o in n.values() for v in o if v != "main"})
out = ["| test | op | variant | runs main/variant | main ns (per-run medians) | variant ns (per-run medians) | change ns | change % | spread ns | verdict |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for node in order:
    for op, vv in data[node].items():
        if only and op not in only or "main" not in vv:
            continue
        m_runs = vv["main"]; m = statistics.median(m_runs)
        for v in variants:
            if v not in vv:
                continue
            c_runs = vv[v]; c = statistics.median(c_runs)
            spread = max(max(m_runs) - min(m_runs), max(c_runs) - min(c_runs))
            verdict = "slower" if c - m > spread else ("faster" if m - c > spread else "equal")
            out.append(f"| {node} | {op} | {v} | {len(m_runs)}/{len(c_runs)} | {m:.0f} ({', '.join(f'{x:.0f}' for x in m_runs)}) | "
                       f"{c:.0f} ({', '.join(f'{x:.0f}' for x in c_runs)}) | {c - m:+.0f} | {100 * (c - m) / m:+.2f} | {spread:.0f} | {verdict} |")
txt = "\n".join(out)
print(txt)
if md:
    open(md, "w").write(txt + "\n")
