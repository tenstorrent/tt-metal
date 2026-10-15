#!/usr/bin/env python3
"""Table of a run_pu_init.sh log: per init form and marker, every run's L1_TO_L1 and PACK_ISOLATE mean and the mean per block
(mean / (loop_factor * num_blocks)), the run spread. usage: pu_table.py <log>"""
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
        for r in csv.DictReader(io.StringIO("\n".join(buf))):
            n = float(r.get("loop_factor") or 1) * float(r.get("num_blocks") or r.get("tile_cnt") or 1)
            for rt in ("L1_TO_L1", "PACK_ISOLATE"):
                v = r.get(f"mean({rt})")
                if v not in (None, ""):
                    vals[(r["pack_untilize_init"], r["marker"], rt)].append((cur, float(v), float(v) / n))
        cur = None
    elif cur:
        buf.append(l)
print("| init | marker | run type | runs | per block (min / max) | spread |")
print("|---|---|---|---|---|---|")
for (init, marker, rt), xs in sorted(vals.items()):
    pb = [x[2] for x in xs]
    print(f"| {init} | {marker} | {rt} | {len(xs)} | {statistics.median(pb):.2f} ({min(pb):.2f} / {max(pb):.2f}) | {max(pb) - min(pb):.2f} |")
