# Effective AICLK of each traced replay: the device profiler's cycle span of the replay (first kernel start to last
# kernel end over all cores and ops of one trace execution) against the host's wall time for that iteration (from the
# run's "Iteration i: ... ms" / QWEN_ITER_TIMING lines). Tells clock throttling apart from host / dispatch gaps: a
# replay that spans N cycles in W ms ran at N / W, and the nominal-clock device time is N / 1.35 GHz.
# Usage: replay_clock.py <profiler dir> <e2e log>
import collections
import csv
import re
import sys

d, log = sys.argv[1], sys.argv[2]
f = open(f"{d}/.logs/profile_log_device.csv")
next(f)
r = csv.reader(f)
ix = {h.strip(): i for i, h in enumerate(next(r))}
lo, hi, ops = collections.defaultdict(lambda: 1 << 62), collections.defaultdict(int), collections.defaultdict(set)
for row in r:
    if not row[ix["zone name"]].strip().endswith("-KERNEL"):
        continue
    tid = row[ix["trace id"]].strip()
    if not tid:
        continue
    key = (int(tid), int(row[ix["trace id counter"]].strip() or 0))
    t = int(row[ix["time[cycles since reset]"]])
    lo[key], hi[key] = min(lo[key], t), max(hi[key], t)
    ops[key].add(row[ix["run host ID"]].strip())
txt = open(log, errors="ignore").read()
iters = [float(v) for v in re.findall(r"Iteration \d+: ([0-9.]+)ms", txt)]
syncs = [float(v) for v in re.findall(r"sync=([0-9.]+) to_torch", txt)]
# the largest trace (most ops) is the model replay; its executions in order
big = max({k[0] for k in lo}, key=lambda t: max(len(ops[k]) for k in ops if k[0] == t))
runs = sorted(k for k in lo if k[0] == big)
print(f"trace {big}: {len(runs)} executions, {len(ops[runs[0]])} ops each; host iterations (ms) {iters}")
for i, k in enumerate(runs):
    span = hi[k] - lo[k]
    nominal = span / 1.35e6
    line = f"execution {i}: span {span} cycles = {nominal:.1f} ms at 1.35 GHz"
    j = len(iters) - len(runs) + i  # the last len(runs) iterations are the profiled replays
    if 0 <= j < len(iters):
        line += f"; host iteration {iters[j]:.1f} ms -> effective clock {span / iters[j] / 1e3:.0f} MHz (if the span fills the iteration)"
    print("RES " + line)
