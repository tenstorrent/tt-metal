"""Steady-state PP throughput from a prefill_runner log: per-rank chunk retirement intervals.

The producer's own tok/s counts its H2D pushes, which include pipeline fill and the first chunk's
kernel compile (visible as a p99 push outlier), so it understates steady state. The honest number is
the LAST rank's chunk-to-chunk interval after the pipeline is full: in steady state one chunk retires
per interval, so throughput = chunk_size / median(interval).
"""
import re
import statistics
import sys

CHUNK = 5120
log = sys.argv[1]
warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 4

starts = {}
pat = re.compile(r"\[pp rank (\d)\] CHUNK_START c=(\d+) compute_start=([0-9.]+)")
for line in open(log, errors="replace"):
    m = pat.search(line)
    if m:
        starts.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3))))

for rank in sorted(starts):
    xs = [t for _, t in sorted(starts[rank])]
    d = [b - a for a, b in zip(xs, xs[1:])]
    if len(d) <= warmup:
        print(f"rank {rank}: only {len(d)} intervals, need > {warmup}")
        continue
    ss = d[warmup:]
    med = statistics.median(ss)
    print(
        f"rank {rank}: n={len(xs)} chunks | steady intervals n={len(ss)} "
        f"median={med*1000:7.1f}ms mean={statistics.mean(ss)*1000:7.1f}ms "
        f"min={min(ss)*1000:7.1f}ms max={max(ss)*1000:7.1f}ms "
        f"-> {CHUNK/med:8.0f} tok/s (median)"
    )
last = max(starts)
xs = [t for _, t in sorted(starts[last])]
print(
    f"\nlast rank ({last}) span: {xs[-1]-xs[0]:.3f}s over {len(xs)-1} intervals "
    f"-> {(len(xs)-1)*CHUNK/(xs[-1]-xs[0]):.0f} tok/s incl. fill"
)
