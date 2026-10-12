#!/usr/bin/env python3
# ra2 part A rule (fixed before the run): Int32 column SUM, TILE_LOOP totals per run. A pacing k qualifies when its
# median is at most main's median + 0.5 cycles in L1_TO_L1 at loop factors 16 and 128 and in MATH_ISOLATE at both;
# qualifying k are ranked by the L1_TO_L1 per-tile sum (lf16 + lf128). Prints the best two (or none) and a table.
# usage: sel_k.py <tag prefix> <runs dir>
import csv, glob, os, statistics, sys
R = sys.argv[2]
pre = sys.argv[1]
vals = {}
for d in glob.glob(f"{R}/{pre}_*"):
    v = os.path.basename(d)[len(pre) + 1:].rsplit("_", 1)[0]
    for fn in glob.glob(f"{d}/perf_sfpu_reduce/perf_sfpu_reduce.csv"):
        for r in csv.DictReader(open(fn)):
            if r["marker"] != "TILE_LOOP" or "Int32" not in r["formats.input_A"] or "Column" not in r["mathop"] or "Sum" not in r["pool_type"]:
                continue
            lf = int(r["loop_factor"])
            for rt in ("L1_TO_L1", "MATH_ISOLATE"):
                x = r.get(f"mean({rt})")
                if x: vals.setdefault(v, {}).setdefault((rt, lf), []).append(float(x))
med = {v: {k: statistics.median(x) for k, x in d.items()} for v, d in vals.items()}
m = med["m"]
rows, ok = [], []
for v in sorted(med, key=lambda s: (s[0], int(s[1:]) if s[1:].isdigit() else 0)):
    d = med[v]
    if any(k not in d for k in m): continue
    q = all(d[k] <= m[k] + 0.5 for k in m)
    score = d[("L1_TO_L1", 16)] / 16 + d[("L1_TO_L1", 128)] / 128
    rows.append(f"{v}\t" + "\t".join(f"{d[k]:.1f}({d[k]-m[k]:+.1f})" for k in sorted(m)) + f"\t{score:.3f}\t{'ok' if q else '-'}\tn={min(len(vals[v][k]) for k in m)}")
    if q and v.startswith("p"): ok.append((score, v))
print("variant\t" + "\t".join(f"{a}/{b}" for a, b in sorted(m)) + "\tscore\tqualifies")
print("\n".join(rows))
best = [v for _, v in sorted(ok)[:2]]
print("BEST " + (" ".join(best) if best else "none"))
