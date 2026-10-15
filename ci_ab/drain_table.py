#!/usr/bin/env python3
"""Table of a run_drain_ab.sh log: per marker, init form and run type, the per block mean (mean / (loop_factor * num_blocks))
of every variant's runs (tag pud_<j>_<variant>_<form>), the median per variant, B vs A, A2 vs A (A/A) and B2 vs B.
usage: drain_table.py <log>"""
import csv, io, re, statistics, sys
from collections import defaultdict
vals = defaultdict(list)
cur, buf = None, []
for l in open(sys.argv[1], errors="ignore"):
    l = re.sub(r"^\S+Z ", "", l.rstrip("\n"))
    m = re.match(r"== csv (\S+)$", l)
    if m:
        cur, buf = m.group(1), []
    elif l == "== end csv" and cur:
        var = cur.split("_")[2]
        for r in csv.DictReader(io.StringIO("\n".join(buf))):
            n = float(r.get("loop_factor") or 1) * float(r.get("num_blocks") or r.get("tile_cnt") or 1)
            for rt in ("L1_TO_L1", "PACK_ISOLATE"):
                v = r.get(f"mean({rt})")
                if v not in (None, ""):
                    vals[(r["marker"], r["pack_untilize_init"], rt, var)].append(float(v) / n)
        cur = None
    elif cur:
        buf.append(l)
med = lambda k: statistics.median(vals[k]) if vals[k] else float("nan")
pc = lambda x, y: 100.0 * (x - y) / y
print("| marker | init | run type | runs | A | B | B vs A % | A2 vs A % (A/A) | B2 vs B % | A min / max | B min / max |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (mk, init, rt) in sorted({k[:3] for k in vals}):
    g = lambda v: vals[(mk, init, rt, v)]
    a, b, a2, b2 = (med((mk, init, rt, v)) for v in ("A", "B", "A2", "B2"))
    rng = lambda xs: f"{min(xs):.2f} / {max(xs):.2f}" if xs else "-"
    print(f"| {mk} | {init} | {rt} | {len(g('A'))},{len(g('B'))},{len(g('A2'))},{len(g('B2'))} | {a:.2f} | {b:.2f} | {pc(b, a):+.2f} | {pc(a2, a):+.2f} | {pc(b2, b):+.2f} | {rng(g('A'))} | {rng(g('B'))} |")
