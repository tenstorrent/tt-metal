#!/usr/bin/env python3
"""Parse a tools/mm_sweep/run_sweep.sh results dir into a main-baseline-vs-branch comparison.

baseline = best block over the explicit sweep (pure unicast == main); branch = default auto path.
Median is taken over reps (dropping the first 3 warmup reps when >6 are present).

Usage: python tools/mm_sweep/parse_sweep.py <results_dir> <shapes_file> [out_md]
"""
import csv
import glob
import math
import re
import statistics
import sys
from collections import defaultdict

RES = sys.argv[1]
SHAPES = sys.argv[2]
OUTMD = sys.argv[3] if len(sys.argv) > 3 else None

ATTR = re.compile(r"M_block_size=(\d+);K_block_size=(\d+);N_block_size=(\d+);subblock_h=(\d+);subblock_w=(\d+)")


def _durations_by_cfg(d):
    fs = sorted(glob.glob(f"{d}/reports/*/ops_perf_results*.csv"))
    if not fs:
        return None
    by = defaultdict(list)
    for r in csv.DictReader(open(fs[-1])):
        v = r.get("DEVICE KERNEL DURATION [ns]", "").strip()
        if not v:
            continue
        m = ATTR.search(r.get("ATTRIBUTES", ""))
        key = f"{m.group(1)}/{m.group(2)}/{m.group(3)} sb{m.group(4)}{m.group(5)}" if m else "default"
        by[key].append(float(v))
    return by


def _med(vals):
    v = vals[3:] if len(vals) > 6 else vals
    return statistics.median(v) if v else None


def best_baseline(tag):
    by = _durations_by_cfg(f"{RES}/baseline_{tag}")
    if not by:
        return None, None
    meds = {c: _med(x) for c, x in by.items() if _med(x) is not None}
    if not meds:
        return None, None
    c = min(meds, key=meds.get)
    return meds[c], c


def branch_time(tag):
    by = _durations_by_cfg(f"{RES}/branch_{tag}")
    if not by:
        return None
    # branch default path emits a single config (or "default"); take the longest list / median.
    vals = max(by.values(), key=len)
    return _med(vals)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


shapes = [tuple(l.split()) for l in open(SHAPES) if l.strip() and not l.startswith("#")]
rows, sp = [], []
for M, K, N in shapes:
    tag = f"{M}x{K}x{N}"
    base, blk = best_baseline(tag)
    br = branch_time(tag)
    if base and br:
        s = base / br
        sp.append(s)
        rows.append((tag, base / 1000, br / 1000, s, blk))
    else:
        rows.append((tag, base / 1000 if base else None, br / 1000 if br else None, None, blk))

rows.sort(key=lambda r: (-(r[3] or 0)))
ok = [r for r in rows if r[3] is not None]
miss = [r for r in rows if r[3] is None]

hdr = (
    f"# minimal_matmul: main optimized baseline vs branch ({RES})\n\n"
    f"baseline = best swept block, pure unicast (TT_MM_NO_LARGE_LEVERS, == main); branch = auto path.\n\n"
)
if sp:
    hdr += (
        f"**geomean {geomean(sp):.3f}x** | wins(>1.05) {sum(1 for x in sp if x > 1.05)}/{len(sp)} | "
        f"within +/-5% {sum(1 for x in sp if 0.95 <= x <= 1.05)} | losses(<0.95) {sum(1 for x in sp if x < 0.95)} | "
        f"min {min(sp):.2f}x max {max(sp):.2f}x\n\n"
    )
tbl = "| shape | baseline us (block) | branch us | speedup |\n|---|--:|--:|:--:|\n"
for tag, b, br, s, blk in rows:
    bcol = f"{b:.1f} ({blk})" if b is not None else "FAIL"
    brcol = f"{br:.1f}" if br is not None else "FAIL"
    tbl += f"| {tag} | {bcol} | {brcol} | {('%.2fx' % s) if s else '-'} |\n"
if miss:
    tbl += f"\n> {len(miss)} shape(s) missing data (check logs): {', '.join(r[0] for r in miss)}\n"

out = hdr + tbl
print(out)
if OUTMD:
    open(OUTMD, "w").write(out)
    print(f"\nwritten to {OUTMD}")
