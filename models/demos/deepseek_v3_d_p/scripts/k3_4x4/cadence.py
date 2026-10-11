"""Throughput from runner timing CSVs (slot, chunk, start_s, compute_ms).
usage: cadence.py <run dir> [chunk_tokens=5120] [skip_first=2]
Steady-state tok/s = chunk_tokens / mean chunk-start interval on the LAST rank, after skip_first chunks."""
import csv
import glob
import statistics as st
import sys

d = sys.argv[1]
tok = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 2
files = sorted(glob.glob(f"{d}/timing/rank*.csv"))
for f in files:
    rows = [(int(r[1]), float(r[2]), float(r[3])) for r in csv.reader(open(f)) if r]
    comp = [c for _, _, c in rows[skip:]]
    print(
        f"{f.split('/')[-1]}: chunks {len(rows)}, compute mean {st.mean(comp):.1f} ms (min {min(comp):.1f} max {max(comp):.1f}) over {len(comp)}"
    )
rows = [(int(r[1]), float(r[2])) for r in csv.reader(open(files[-1])) if r][skip:]
iv = [b[1] - a[1] for a, b in zip(rows, rows[1:])]
print(f"cadence (last rank) {1e3*st.mean(iv):.1f} ms -> {tok/st.mean(iv):,.0f} tok/s over {len(iv)} intervals")
