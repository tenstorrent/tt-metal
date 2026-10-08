# Round r01 summary (policy v0, W=4, R=4)

16 attempts in 4 steps, all committed and verified. 14 valid, 0 hard failures, 0 lost.
Baseline (µs, per-chip mean): 16.99 / 18.26 / 23.48 / 26.03. Noise ±1%.

## Best
r01-b04-a04, score 1.2132: 14.17 / 15.43 / 19.08 / 20.98 µs (1.199 / 1.183 / 1.231 / 1.241x). PCC unchanged (0.9999985).
Lineage: x*gamma under the AG wait -> trid-pipelined input + gamma interleave (regressed) -> gamma on the idle BRISC -> DST-accumulated sum(x^2), one pack per row.

## Grid (score)
| branch | a01 | a02 | a03 | a04 |
|---|---|---|---|---|
| b01 | 1.109 | 0.900 | 1.187 | 1.204 |
| b02 | 1.003 | 1.034 | 1.172 | 0.985 |
| b03 | 0.987 | 1.094 | 1.131 | 1.165 |
| b04 | 1.076 | 0.840 | 1.208 | 1.213 |

## What worked
- Overlap x*gamma with the all-gather wait (found independently by b01 and b04 in step 1).
- Read gamma on the idle writer RISC at kernel start, keeping the reader on a trid-pipelined input path (b01-a03, b04-a03).
- Accumulate sum(x^2) in DST and pack once per row (b01-a04, b04-a04).
- Column split with equal-width slices + DRAM bank de-phasing + dual-NoC output drain (b03 line, best on h7168 relative to its own lineage).

## What failed and why
- Interleaving gamma reads with the input stream (b01-a02, b04-a02): gamma is a shared-bank hot spot, so it serializes the critical path. Repaired a step later by moving it to BRISC.
- Dual-NoC drain on the b02 lineage (b02-a04): NoC1 hot spot on right-half cores; same idea helped on b03. Needs a per-core, position-aware split.
- First column-split attempts: packet cap (b02-a01) and uneven slices -> two kernel groups -> dispatch stall (b03-a01). Both repaired.

## Policy observations (input for dreaming)
- Repairs paid off: three branches dipped (0.90, 0.84, 0.987) and the next attempt beat the parent. A policy that closes on one bad result would have lost the two best nodes.
- Step-1 workers converged on the same idea (b01/b04) because they started with empty history at the same time.
- Same-step workers cannot see each other: b01-a02 and b04-a02 made the same mistake in parallel.
- The column-split line and the gamma/DST line are complementary and were never combined (cross-branch merges are not allowed inside a round). Natural round-2 root candidate: r01-b04-a04, with the column split as the obvious port.

## Ops
- One worker (r01-b02-a03) was cut off by a launcher bug after its eval; resumed and committed. Launcher fixed (node tag = completion).
- Approximate worker cost for the round: ~$45.
- v0 replay on r01: V = 1.1732 (best 1.213, 16 attempts, 4 steps).
