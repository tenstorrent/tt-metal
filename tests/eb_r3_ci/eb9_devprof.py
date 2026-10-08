#!/usr/bin/env python3
"""#58724 fourth review: device time of fused programs from the device profiler's device-side log (slow dispatch, no host capture).
Reads <root>/<test>/<pass>_<side>/.logs/profile_log_device.csv. Slow dispatch starts the cores of a program one after another, so
program spans measure the launch; the per-core kernel duration does not: for every chip, core and Tensix RISC (BRISC, NCRISC,
TRISC_0..2) the duration of its *-KERNEL zone, the longest one of the run (the fused program; the setup programs are short). Per
core RISC the sides compare as pooled medians over the runs with the spread (the larger range of the two sides); 'slower' marks
a change above the spread. A chip's busy time is its longest core RISC duration; the device time of a run is the longest busy time.
With --dump each run's per-core durations are also printed as one gzip+base64 line (DUMP <test> <run> <blob>).
usage: eb9_devprof.py <root> [side A] [side B] [--dump]"""
import base64
import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = args[0]
A = args[1] if len(args) > 1 else "head"
B = args[2] if len(args) > 2 else "pr"
DUMP = "--dump" in sys.argv
TENSIX = ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2")


def parse(path):
    """{(chip, x, y, risc): longest KERNEL zone duration in ns}"""
    with open(path) as f:
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", f.readline())
        mhz = float(m.group(1)) if m else 1350.0
        rd = csv.reader(f)
        cols = [c.strip() for c in next(rd)]
        ix = {c: i for i, c in enumerate(cols)}
        open_at, best = {}, defaultdict(float)
        for r in rd:
            if len(r) < len(cols):
                continue
            zone, risc = r[ix["zone name"]].strip(), r[ix["RISC processor type"]].strip()
            if not zone.endswith("-KERNEL") or risc not in TENSIX:
                continue
            core = (r[ix["PCIe slot"]].strip(), r[ix["core_x"]].strip(), r[ix["core_y"]].strip(), risc)
            key = core + (r[ix["run host ID"]].strip(),)
            t = int(r[ix["time[cycles since reset]"]])
            typ = r[ix["type"]].strip()
            if typ == "ZONE_START":
                open_at[key] = t
            elif typ == "ZONE_END" and key in open_at:
                best[core] = max(best[core], (t - open_at.pop(key)) * 1000.0 / mhz)
    return dict(best)


def verdict(a, b):
    ma, mb = statistics.median(a), statistics.median(b)
    sp = max(max(a) - min(a), max(b) - min(b))
    return ma, mb, sp, ("slower" if mb - ma > sp else ("faster" if ma - mb > sp else "equal"))


for tdir in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(tdir):
        continue
    test = os.path.basename(tdir)
    cores = defaultdict(lambda: {A: [], B: []})
    busy = defaultdict(lambda: {A: [], B: []})
    dev = {A: [], B: []}
    runs = {A: 0, B: 0}
    for rdir in sorted(glob.glob(os.path.join(tdir, "*"))):
        side = os.path.basename(rdir).split("_", 1)[1]
        f = glob.glob(os.path.join(rdir, ".logs", "profile_log_device.csv"))
        if not f:
            continue
        c = parse(f[0])
        if DUMP:
            blob = base64.b64encode(gzip.compress(json.dumps(sorted([list(k) + [round(v, 1)] for k, v in c.items()])).encode())).decode()
            print(f"DUMP {test} {os.path.basename(rdir)} {blob}")
        if side not in dev or not c:
            continue
        runs[side] += 1
        per_chip = defaultdict(float)
        for core, ns in c.items():
            cores[core][side].append(ns)
            per_chip[core[0]] = max(per_chip[core[0]], ns)
        for chip, ns in per_chip.items():
            busy[chip][side].append(ns)
        dev[side].append(max(per_chip.values()))
    print(f"== {test}: runs {A} {runs[A]}, {B} {runs[B]}, chips {len(busy)}")
    if not dev[A] or not dev[B]:
        continue
    for chip, v in sorted(busy.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        if len(v[A]) and len(v[B]):
            ma, mb, sp, vd = verdict(v[A], v[B])
            print(f"   chip {chip} busy: {A} {ma:.0f} ns, {B} {mb:.0f} ns, change {mb - ma:+.0f} ns ({100 * (mb - ma) / ma:+.4f} %), spread {sp:.0f} ns, {vd}")
    ma, mb, sp, vd = verdict(dev[A], dev[B])
    print(f"   device time (longest core RISC per run): {A} {ma:.0f} ns, {B} {mb:.0f} ns, change {mb - ma:+.0f} ns ({100 * (mb - ma) / ma:+.4f} %), spread {sp:.0f} ns, {vd}")
    rows, cnt, tot = [], defaultdict(lambda: defaultdict(int)), defaultdict(lambda: [0.0, 0.0])
    for core, v in cores.items():
        if len(v[A]) != runs[A] or len(v[B]) != runs[B]:
            continue
        ma, mb, sp, vd = verdict(v[A], v[B])
        cnt[core[3]][vd] += 1
        tot[core[3]][0] += ma
        tot[core[3]][1] += mb
        rows.append((mb - ma, ma, mb, sp, vd, core))
    if not rows:
        continue
    d = sorted(r[0] for r in rows)
    q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
    print(f"   per core RISC ({len(rows)} in every run): change ns min {d[0]:+.0f}, p10 {q(0.1):+.0f}, median {q(0.5):+.0f}, p90 {q(0.9):+.0f}, max {d[-1]:+.0f}")
    for risc in sorted(cnt):
        c = cnt[risc]
        print(f"   {risc}: slower/equal/faster {c['slower']}/{c['equal']}/{c['faster']}; summed {A} {tot[risc][0]:.0f} ns, {B} {tot[risc][1]:.0f} ns ({100 * (tot[risc][1] - tot[risc][0]) / tot[risc][0]:+.4f} %)")
    long = sorted(rows, key=lambda r: -r[1])[:8]
    for r in long:
        print(f"   longest: chip {r[5][0]} core ({r[5][1]},{r[5][2]}) {r[5][3]}: {A} {r[1]:.0f} ns, {B} {r[2]:.0f} ns, {r[0]:+.0f} ns, spread {r[3]:.0f}, {r[4]}")
    for r in sorted([r for r in rows if r[4] == "slower"], key=lambda r: -r[0])[:10]:
        print(f"   slower: chip {r[5][0]} core ({r[5][1]},{r[5][2]}) {r[5][3]}: {A} {r[1]:.0f} ns, {B} {r[2]:.0f} ns, {r[0]:+.0f} ns, spread {r[3]:.0f}")
    for r in sorted([r for r in rows if r[4] == "faster"], key=lambda r: r[0])[:10]:
        print(f"   faster: chip {r[5][0]} core ({r[5][1]},{r[5][2]}) {r[5][3]}: {A} {r[1]:.0f} ns, {B} {r[2]:.0f} ns, {r[0]:+.0f} ns, spread {r[3]:.0f}")
