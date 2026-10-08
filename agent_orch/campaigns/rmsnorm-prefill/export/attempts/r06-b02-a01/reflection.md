# r06-b02-a01 result: 1.5419 (ok)

## What happened vs expected
The result is valid on every shape on the first run, with no hang. PCC is 0.9999985 and max_abs 0.0205-0.0240, the
parent's values. So the uneven 9/11 split itself is correct:
- the Bresenham wave interleave, rows and slots;
- the double-partner start sem;
- the per-wave forwarder thresholds;
- the 22-slot / 3456 B wave span.

Score 1.5419 vs parent r05-b01-a01 1.5696 (**-1.8%**). µs, chip mean:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.19 | 11.43 | +2.1% (worse) |
| h4096 | 12.08 | 12.39 | +2.6% (worse) |
| h6144 | 14.59 | 15.00 | +2.8% (worse) |
| h7168 | 15.84 | 15.79 | -0.3% (noise) |

I expected -0.3..-0.7 µs on the wide shapes. It was neutral on h7168 and worse everywhere else.

## Why (profiler evidence)
I ran the parent's `analysis/waves.py` and `fwd.py` on this node's report. Outputs are in `analysis/`. Values are
medians of per-call maxima, in µs from chip start, shown as this / parent.

1. **The model's premise is wrong: wave A's read time doesn't scale with wave A's row count.**
   - A read end: 3.53 / 3.59 (h7168), 3.22 / 3.15 (h6144), 2.09 / 2.08 (h3584).
   - Cutting A from 20 to 18 workers moved nothing. Each worker reads exactly one half-row (28 tiles at h7168,
     about 17 GB/s per core), so A's read end is a **per-core** time (read depth / latency), not the wave's
     aggregate volume over DRAM bandwidth.
   - A smaller A therefore doesn't start its chain earlier. A push 4.39 / 4.51 and F_SEND#0 end 5.14 / 5.51
     improved a little, from fewer arrivals to wait for. But F_GO#0 start 7.87 / 7.95 barely moved, because the AG
     under load is ~2.7 µs.
2. **Wave B got bigger, so it got later.** B has 22 workers instead of 20, and its read is the one that runs at
   the aggregate rate after A's tail.
   - B read end: 6.58 / 6.29 (h7168), 5.82 / 5.48 (h6144), 3.61 / 3.43 (h3584).
   - B go and drain start move later by the same 0.2-0.5 µs: B drain_s 11.57 / 11.31 and 11.20 / 10.72.
3. **The drain-overlap part of the model did work.** At h7168, A's drain ends at 13.20 vs 14.21 because it is less
   contended. B's drain is shorter (3.64 vs 3.80 med), and the kernel end is 15.67 / 15.74. At h7168 that gain just
   cancels B's later start.
   - On the narrower shapes A's drain was not the problem: A drain_e 12.89 / 12.95 at h6144. B's later chain is a
     pure loss there.
4. A side lesson: in the 2-wave design the kernel end ≈ B's chain end (B read end + ~5 µs) + B's drain. B's read
   end is set by total read volume, so it can't move by rebalancing rows. Only a faster chain or drain helps.

## Classification
flawed idea (for this direction of skew). The wave-A read is per-core bound, not aggregate bound, so moving rows
from A to B only lengthens B. The plumbing is correct and reusable for other ratios.

## What a child of this node should try next
1. **Don't move rows from A to B.** If any skew is tried, go the other way (A 11 / B 9: `dit_rmsnorm_wave_a_rows`
   returns R*11/20). The factory supports it only if every A worker gets ≥1 B partner, which needs a mapping
   tweak: A ranks without a partner must use role 0. A's read end won't move (it's per-core), B's read ends
   earlier (fewer B cores), and B's chain + drain are the tail. Then A's drain becomes the risk.
   This is r05-b03-a01 #3's suggestion; this node's data supports it.
2. **The per-core read rate is the real lever for wave A** (~17 GB/s/core, read end 3.5 µs at h7168). Try a deeper
   per-core input pipeline on the col-split path only (`kInputLookahead` 4 -> 6-8 in the reader, trids allow up to
   13). r04-b04-a03 tested deeper reads on the 20-full-row design, where the read was aggregate-bound. Here A's
   read is not, so the result can differ.
3. The obvious port is still open on this lineage: the writer un-gate (r05-b01-a03 #1), and B's go/drain chain.
   F_GO is ~0.55 µs, go -> drain 0.5 µs.
