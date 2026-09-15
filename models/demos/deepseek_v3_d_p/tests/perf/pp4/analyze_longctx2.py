"""Long-context chunked PP throughput in EXACTLY the two metrics MISTRAL4_PREFILL_PERFORMANCE.md
§9 reports for test_mistral4_pp4_concurrent_longctx, so the numbers are directly comparable:

  total  = CONTEXT / total_wall
  steady = CHUNK / median(per_chunk_interval[PP-1:])     # full-pipeline iterations only

`total_wall` here is rank 0's first compute start to the last rank's last compute end (the runner's
own E2E_CLOCK epochs), which is the multi-process analogue of the single-process loop's summed dt.
"""
import re
import statistics
import sys

log, CHUNK, PP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cs = re.compile(r"\[pp rank (\d)\] CHUNK_START c=(\d+) compute_start=([0-9.]+) slot=(-?\d+) \[(\d+),(\d+)\)")
e2e = re.compile(r"\[pp rank (\d)\] E2E_CLOCK first_compute_start=([0-9.na/]+) last_compute_end=([0-9.]+)")

rows, clocks = {}, {}
for line in open(log, errors="replace"):
    m = cs.search(line)
    if m:
        rows.setdefault(int(m[1]), []).append((int(m[2]), float(m[3]), int(m[5]), int(m[6])))
    m = e2e.search(line)
    if m:
        try:
            clocks[int(m[1])] = (float(m[2]), float(m[3]))
        except ValueError:
            pass

last = max(rows)
seq = sorted(rows[last])
reqs, cur = [], []
for rec in seq:
    if rec[2] == 0 and cur:
        reqs.append(cur)
        cur = []
    cur.append(rec)
if cur:
    reqs.append(cur)

tokens = sum(r[-1][3] for r in reqs)
if 0 in clocks and last in clocks:
    wall = clocks[last][1] - clocks[0][0]
    print(f"whole run: {len(reqs)} request(s), {tokens} tokens, wall {wall:.3f}s -> TOTAL {tokens/wall:,.0f} tok/s")

print(f"\nper request (steady = CHUNK/median(intervals[{PP-1}:]), §9's definition):")
for i, r in enumerate(reqs):
    ts = [t for _, t, _, _ in r]
    d = [b - a for a, b in zip(ts, ts[1:])]
    if len(d) <= PP - 1:
        print(f"  request {i}: too few intervals")
        continue
    full = d[PP - 1 :]
    med = statistics.median(full)
    print(
        f"  request {i}: window={r[-1][3]} chunks={len(r)} | steady median {med*1000:.1f} ms/chunk "
        f"-> STEADY {CHUNK/med:,.0f} tok/s  (deepest chunk {d[-1]*1000:.1f} ms -> {CHUNK/d[-1]:,.0f} tok/s)"
    )
