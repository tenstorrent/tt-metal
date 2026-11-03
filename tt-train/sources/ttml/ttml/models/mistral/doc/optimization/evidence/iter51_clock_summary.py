"""Summarise iter51_telemetry_sampler.py's CSV: the step phase is the longest stretch where the mean TDP over the
chips stays above 100 W (the loading phase sits at 60-85 W); per chip the AICLK mean / min / max, the mean TDP, the
peak temperature and the throttler masks seen inside it; the MFU the log reports rescaled to the delivered clock."""
import csv
import re
import sys
from collections import Counter, defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
by_t = defaultdict(dict)
for r in rows:
    by_t[float(r["t"])][int(r["chip"])] = r
times = sorted(by_t)
hot = [t for t in times if len(by_t[t]) == 4 and sum(float(by_t[t][c]["tdp"]) for c in range(4)) / 4 > 100]
# longest contiguous (< 0.5 s gaps) stretch of hot samples
best, cur = [], []
for t in hot:
    if cur and t - cur[-1] > 0.5:
        if len(cur) > len(best):
            best = cur
        cur = []
    cur.append(t)
if len(cur) > len(best):
    best = cur
if not best:
    print("no step phase found (mean TDP never above 100 W)")
    sys.exit(0)
print(
    f"step phase: {best[-1] - best[0]:.1f} s, {len(best)} samples at {len(best) / max(best[-1] - best[0], 1e-9):.0f} Hz"
)
clocks = {}
for c in range(4):
    ai = [int(by_t[t][c]["aiclk"]) for t in best]
    tdp = [float(by_t[t][c]["tdp"]) for t in best]
    tdc = [float(by_t[t][c]["tdc"]) for t in best]
    temp = [float(by_t[t][c]["temp"]) for t in best]
    thr = Counter(by_t[t][c]["throttler"] for t in best)
    clocks[c] = sum(ai) / len(ai)
    print(
        f"chip {c}: aiclk mean {clocks[c]:6.0f} MHz (min {min(ai)}, max {max(ai)}), tdp mean {sum(tdp) / len(tdp):5.0f} W"
        f" (max {max(tdp):.0f}), tdc mean {sum(tdc) / len(tdc):4.0f} A, temp max {max(temp):.0f} C, throttler {dict(thr)}"
    )
mfu = None
if len(sys.argv) > 2:
    for line in open(sys.argv[2]):
        m = re.search(r"MFU ([\d.]+)%", line)
        if m:
            mfu = float(m.group(1))
if mfu is not None:
    slowest = min(clocks.values())
    mean = sum(clocks.values()) / 4
    print(
        f"MFU {mfu:.2f} % of the 1350 MHz peak = {mfu * 1350 / slowest:.1f} % at the slowest chip's delivered clock"
        f" ({slowest:.0f} MHz), {mfu * 1350 / mean:.1f} % at the mean delivered clock ({mean:.0f} MHz)"
    )
