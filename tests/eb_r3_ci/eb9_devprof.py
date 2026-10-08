#!/usr/bin/env python3
"""#58724 fourth review: device time of fused programs from the device profiler's device-side log (slow dispatch, no host capture).
Reads <root>/<test>/<pass>_<side>/.logs/profile_log_device.csv. Per chip and program (run host ID) the span is the latest
*-KERNEL zone end minus the earliest *-KERNEL zone start over the Tensix RISCs (BRISC, NCRISC, TRISC_0..2; fabric ERISCs left
out). A chip's fused program is every program whose span is at least half the chip's longest; a run's sample per chip is the
median of those spans. Per test and chip the sides compare as pooled medians over the runs, with the spread (the larger range of
the two sides): 'slower' marks a change above the spread. The device time of a run is the largest span over the chips. Per core
and RISC the kernel duration (its *-KERNEL zone, median over the fused programs of the run) compares the same way; printed are
the counts, the distribution of the changes, the summed duration per RISC type and the cores whose change exceeds the spread.
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


def parse(path):
    with open(path) as f:
        head = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", head)
        mhz = float(m.group(1)) if m else 1350.0
        rd = csv.reader(f)
        cols = [c.strip() for c in next(rd)]
        ix = {c: i for i, c in enumerate(cols)}
        lo, hi, zs, ze = {}, {}, defaultdict(list), defaultdict(list)
        for r in rd:
            if len(r) < len(cols):
                continue
            zone, risc = r[ix["zone name"]].strip(), r[ix["RISC processor type"]].strip()
            if not zone.endswith("-KERNEL") or risc not in TENSIX:
                continue
            key = (r[ix["PCIe slot"]].strip(), r[ix["run host ID"]].strip())
            t = int(r[ix["time[cycles since reset]"]])
            typ = r[ix["type"]].strip()
            core = (key[0], r[ix["core_x"]].strip(), r[ix["core_y"]].strip(), risc)
            if typ == "ZONE_START":
                lo[key] = min(lo.get(key, t), t)
                zs[(key[1],) + core].append(t)
            elif typ == "ZONE_END":
                hi[key] = max(hi.get(key, t), t)
                ze[(key[1],) + core].append(t)
    per_chip = defaultdict(list)
    for key in lo:
        if key in hi and hi[key] > lo[key]:
            per_chip[key[0]].append((hi[key] - lo[key]) * 1000.0 / mhz)
    out, fused = {}, set()
    for chip, v in per_chip.items():
        top = max(v)
        out[chip] = statistics.median([x for x in v if x >= 0.5 * top])
        fused |= {k for k in lo if k[0] == chip and k in hi and (hi[k] - lo[k]) * 1000.0 / mhz >= 0.5 * top}
    cores = defaultdict(list)
    for k, starts in zs.items():
        run, core = k[0], k[1:]
        if (core[0], run) not in fused or len(ze.get(k, [])) != len(starts):
            continue
        cores[core] += [(e - b) * 1000.0 / mhz for b, e in zip(sorted(starts), sorted(ze[k]))]
    return out, {c: statistics.median(v) for c, v in cores.items()}


for tdir in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(tdir):
        continue
    test = os.path.basename(tdir)
    chips = defaultdict(lambda: {A: [], B: []})
    cores = defaultdict(lambda: {A: [], B: []})
    dev = {A: [], B: []}
    runs = {A: 0, B: 0}
    for rdir in sorted(glob.glob(os.path.join(tdir, "*"))):
        side = os.path.basename(rdir).split("_", 1)[1]
        f = glob.glob(os.path.join(rdir, ".logs", "profile_log_device.csv"))
        if side not in dev or not f:
            continue
        s, c = parse(f[0])
        if not s:
            continue
        runs[side] += 1
        for chip, ns in s.items():
            chips[chip][side].append(ns)
        for core, ns in c.items():
            cores[core][side].append(ns)
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
    rows, ccnt, tot = [], defaultdict(lambda: defaultdict(int)), defaultdict(lambda: [0.0, 0.0])
    for core, v in cores.items():
        if len(v[A]) != runs[A] or len(v[B]) != runs[B]:
            continue
        ma, mb = statistics.median(v[A]), statistics.median(v[B])
        sp = max(max(v[A]) - min(v[A]), max(v[B]) - min(v[B]))
        verdict = "slower" if mb - ma > sp else ("faster" if ma - mb > sp else "equal")
        ccnt[core[3]][verdict] += 1
        tot[core[3]][0] += ma
        tot[core[3]][1] += mb
        rows.append((mb - ma, ma, mb, sp, verdict, core))
    if rows:
        d = sorted(r[0] for r in rows)
        q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
        print(f"   per-core kernel durations: {len(rows)} core RISCs in every run; change ns min {d[0]:+.0f}, p10 {q(0.1):+.0f}, median {q(0.5):+.0f}, p90 {q(0.9):+.0f}, max {d[-1]:+.0f}")
        for risc in sorted(ccnt):
            c = ccnt[risc]
            print(f"   {risc}: slower/equal/faster {c['slower']}/{c['equal']}/{c['faster']}; summed duration {A} {tot[risc][0]:.0f} ns, {B} {tot[risc][1]:.0f} ns ({100 * (tot[risc][1] - tot[risc][0]) / tot[risc][0]:+.4f} %)")
        for r in sorted([r for r in rows if r[4] == "slower"], key=lambda r: -r[0])[:12]:
            print(f"   slower: chip {r[5][0]} core ({r[5][1]},{r[5][2]}) {r[5][3]}: {A} {r[1]:.0f} ns, {B} {r[2]:.0f} ns, {r[0]:+.0f} ns, spread {r[3]:.0f}")
        for r in sorted([r for r in rows if r[4] == "faster"], key=lambda r: r[0])[:6]:
            print(f"   faster: chip {r[5][0]} core ({r[5][1]},{r[5][2]}) {r[5][3]}: {A} {r[1]:.0f} ns, {B} {r[2]:.0f} ns, {r[0]:+.0f} ns, spread {r[3]:.0f}")
