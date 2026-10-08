#!/usr/bin/env python3
"""#58724 fourth review: device time of fused programs from the device profiler's device-side log (slow dispatch, no host capture).
Reads <root>/<test>/<pass>_<side>/.logs/profile_log_device.csv. Per chip and program (run host ID) the span is the latest
*-KERNEL zone end minus the earliest *-KERNEL zone start over the Tensix RISCs (BRISC, NCRISC, TRISC_0..2; fabric ERISCs left
out). A chip's fused program is every program whose span is at least half the chip's longest; a run's sample per chip is the
median of those spans. Per test and chip the sides compare as pooled medians over the runs, with the spread (the larger range of
the two sides): 'slower' marks a change above the spread. The device time of a run is the largest span over the chips.
usage: eb9_devprof.py <root> [side A] [side B]"""
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

root = sys.argv[1]
A = sys.argv[2] if len(sys.argv) > 2 else "head"
B = sys.argv[3] if len(sys.argv) > 3 else "pr"
TENSIX = ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2")


def spans(path):
    with open(path) as f:
        head = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", head)
        mhz = float(m.group(1)) if m else 1350.0
        rd = csv.reader(f)
        cols = [c.strip() for c in next(rd)]
        ix = {c: i for i, c in enumerate(cols)}
        lo, hi = {}, {}
        for r in rd:
            if len(r) < len(cols):
                continue
            zone, risc = r[ix["zone name"]].strip(), r[ix["RISC processor type"]].strip()
            if not zone.endswith("-KERNEL") or risc not in TENSIX:
                continue
            key = (r[ix["PCIe slot"]].strip(), r[ix["run host ID"]].strip())
            t = int(r[ix["time[cycles since reset]"]])
            typ = r[ix["type"]].strip()
            if typ == "ZONE_START":
                lo[key] = min(lo.get(key, t), t)
            elif typ == "ZONE_END":
                hi[key] = max(hi.get(key, t), t)
    per_chip = defaultdict(list)
    for key in lo:
        if key in hi and hi[key] > lo[key]:
            per_chip[key[0]].append((hi[key] - lo[key]) * 1000.0 / mhz)
    out = {}
    for chip, v in per_chip.items():
        top = max(v)
        out[chip] = statistics.median([x for x in v if x >= 0.5 * top])
    return out


for tdir in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(tdir):
        continue
    test = os.path.basename(tdir)
    chips = defaultdict(lambda: {A: [], B: []})
    dev = {A: [], B: []}
    runs = {A: 0, B: 0}
    for rdir in sorted(glob.glob(os.path.join(tdir, "*"))):
        side = os.path.basename(rdir).split("_", 1)[1]
        f = glob.glob(os.path.join(rdir, ".logs", "profile_log_device.csv"))
        if side not in dev or not f:
            continue
        s = spans(f[0])
        if not s:
            continue
        runs[side] += 1
        for chip, ns in s.items():
            chips[chip][side].append(ns)
        dev[side].append(max(s.values()))
    print(f"== {test}: runs {A} {runs[A]}, {B} {runs[B]}, chips {len(chips)}")
    if not dev[A] or not dev[B]:
        continue
    cnt = defaultdict(int)
    rows = []
    for chip, v in sorted(chips.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        if not v[A] or not v[B]:
            continue
        ma, mb = statistics.median(v[A]), statistics.median(v[B])
        sp = max(max(v[A]) - min(v[A]), max(v[B]) - min(v[B]))
        verdict = "slower" if mb - ma > sp else ("faster" if ma - mb > sp else "equal")
        cnt[verdict] += 1
        rows.append(f"   chip {chip}: {A} {ma:.0f} ns, {B} {mb:.0f} ns, change {mb - ma:+.0f} ns ({100 * (mb - ma) / ma:+.3f} %), spread {sp:.0f} ns, {verdict}")
    print("\n".join(rows))
    ma, mb = statistics.median(dev[A]), statistics.median(dev[B])
    sp = max(max(dev[A]) - min(dev[A]), max(dev[B]) - min(dev[B]))
    verdict = "slower" if mb - ma > sp else ("faster" if ma - mb > sp else "equal")
    print(f"   device time (largest chip span per run): {A} {ma:.0f} ns {[round(x) for x in dev[A]]}, {B} {mb:.0f} ns {[round(x) for x in dev[B]]}, change {mb - ma:+.0f} ns ({100 * (mb - ma) / ma:+.3f} %), spread {sp:.0f} ns, {verdict}")
    print(f"   chips slower/equal/faster: {cnt['slower']}/{cnt['equal']}/{cnt['faster']}")
