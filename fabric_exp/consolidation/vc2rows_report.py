#!/usr/bin/env python3
"""vc2rows_report.py TAG [TAG ...]: per-sender cycles/packet of vc2rows_runs.sh runs (labels v2t<TAG>_<router>_<split>_
<row>_<rep>), from the bandwidth profiler's per-sender log lines ("Sender core ... (VCn) ... P packets in C cycles" and
"another sender core ... (VCn) with P packets in C cycles"). For vc02ew4352 VC0 finishes first (its rate is under
contention); for vc02bew4352 VC2 finishes first. Mean over repetitions, with each repetition in parentheses."""
import collections, glob, os, re, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LINE = re.compile(r"(?:Sender core \S+ \(VC(\d)\) on device \d+ direction \S+ link \d+: (\d+) packets in (\d+) cycles"
                  r"|another sender core \S+ \(VC(\d)\) with (\d+) packets in (\d+) cycles)")
data = collections.defaultdict(list)
for tag in sys.argv[1:]:
    for f in sorted(glob.glob(f"{R}/v2t{tag}_*.log")):
        m = re.match(rf"v2t{tag}_(v[^_]+)_([^_]+)_(.+)_(\d+)\.log$", os.path.basename(f))
        if not m:
            continue
        router, split, row, rep = m.groups()
        text = re.sub(r"\x1b\[[0-9;]*m", "", open(f, errors="replace").read())
        for g in LINE.findall(text):
            vc, packets, cycles = (g[0], g[1], g[2]) if g[0] else (g[3], g[4], g[5])
            data[(router, split, row, f"VC{vc}")].append(int(cycles) / int(packets))
print(f"{'router':10} {'split':5} {'row':12} {'sender':6} {'cycles/packet':>14}")
for k in sorted(data):
    v = data[k]
    print(f"{k[0]:10} {k[1]:5} {k[2]:12} {k[3]:6} {sum(v) / len(v):14.1f}  ({' / '.join(f'{x:.1f}' for x in v)})")
