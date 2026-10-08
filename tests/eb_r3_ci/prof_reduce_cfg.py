#!/usr/bin/env python3
"""Round 3 eltwise binary, rules of 16:40 / 17:40: prof_reduce.py per op configuration. Rows keyed by test node, op code and
the op's inputs and output (shape, layout, dtype, memory config) and attributes; per repetition the launches of a key are
summed, per run the median over repetitions (repetition 0 dropped), the table compares each variant with main.
usage: prof_reduce_cfg.py <prof dir> <run tag> [--ops OP1,OP2]"""
import csv, glob, os, statistics, sys, hashlib
from collections import defaultdict

d, tag = sys.argv[1], sys.argv[2]
only = set(sys.argv[sys.argv.index("--ops") + 1].split(",")) if "--ops" in sys.argv else None
COL = "DEVICE KERNEL DURATION [ns]"
data = defaultdict(lambda: defaultdict(list)); count = {}; order = []
def sig(row):
    parts = []
    for p in ("INPUT_0", "INPUT_1", "OUTPUT_0"):
        dims = "x".join(row.get(f"{p}_{a}_PAD[LOGICAL]", "") for a in ("W", "Z", "Y", "X"))
        if dims.strip("x"):
            parts.append(f"{p[0]}{p[-1]}:{dims}:{row.get(p + '_DATATYPE', '')}:{row.get(p + '_MEMORY', '').replace('DEV_0_', '')}")
    h = hashlib.md5(row.get("ATTRIBUTES", "").encode()).hexdigest()[:6]
    return " ".join(parts) + f" attr:{h}"
runs = sorted(glob.glob(os.path.join(d, f"out_{tag}_*_*")), key=lambda p: int(os.path.basename(p).split("_")[2]))
for rd in runs:
    v = os.path.basename(rd).split("_", 3)[3]
    fs = glob.glob(os.path.join(rd, "reports", "*", "ops_perf_results_*.csv"))
    if not fs:
        continue
    cur = None; per = defaultdict(lambda: defaultdict(float)); n = defaultdict(lambda: defaultdict(int))
    for row in csv.DictReader(open(fs[0])):
        if row.get("OP TYPE") == "signpost":
            node, _, k = row["OP CODE"].rpartition("#")
            cur = (node, int(k)) if k.isdigit() else None
            continue
        if cur is None or cur[1] == 0 or (only and row["OP CODE"] not in only):
            continue
        key = (cur[0], row["OP CODE"], sig(row))
        if key not in count:
            order.append(key)
        try:
            per[key][cur[1]] += float(row[COL]); n[key][cur[1]] += 1
        except ValueError:
            pass
    for key, reps in per.items():
        data[key][v].append(statistics.median(reps.values())); count[key] = max(n[key].values())
print("| test | op | config | launches | main ns (per-run medians) | variant ns | change % | spread ns | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for key in order:
    vv = data.get(key, {})
    if "main" not in vv: continue
    for v in sorted(x for x in vv if x != "main"):
        m_r, c_r = vv["main"], vv[v]; m, c = statistics.median(m_r), statistics.median(c_r)
        spread = max(max(m_r) - min(m_r), max(c_r) - min(c_r))
        verdict = "slower" if c - m > spread else ("faster" if m - c > spread else "equal")
        print(f"| {key[0].split('::')[-1][:70]} | {key[1]} | {key[2]} | {count[key]} | {m:.0f} ({', '.join(f'{x:.0f}' for x in m_r)}) | {c:.0f} ({', '.join(f'{x:.0f}' for x in c_r)}) | {(c - m) / m * 100:+.2f} | {spread:.0f} | {verdict} |")
