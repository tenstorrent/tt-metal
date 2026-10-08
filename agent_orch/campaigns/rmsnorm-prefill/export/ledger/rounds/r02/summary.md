# Round r02 summary (policy v1, W=4, R=4, root r01-b04-a04 = 1.2132)

7 attempts in 4 steps (v1 closed 3 of 4 branches after step 1). All committed and verified, all valid.

## Best
r02-b02-a03, score 1.2396: 13.95 / 15.12 / 18.67 / 20.38 µs (1.218 / 1.208 / 1.257 / 1.277x). PCC unchanged.
r02-b02-a01 (1.2385) is equivalent within noise.

## Grid (score)
| branch | a01 | a02 | a03 | a04 |
|---|---|---|---|---|
| b01 | 1.106 (closed) | | | |
| b02 | 1.2385 | 1.2292 | 1.2396 | 1.2346 |
| b03 | 1.031 (closed) | | | |
| b04 | 0.847 (closed) | | | |

## What worked
- Path-aware dual-NoC output drain: only short-eastward DRAM destinations go on NoC0, on alternate bank visits (b02-a01, +2.1% over root).
- Row stat via DST-accumulated matmul (b02-a02/a03): neutral on the per-chip mean, but cut the max-over-chips latency (15.94 -> 14.38 µs at h3584).
- Multicast AG release on a forked forwarder (b02-a04): neutral.

## What failed and why
- Placing the workers in 2 full columns (b04): about 30% slower. The worker classified column stacking as a dead end.
- Dual-NoC drain with a left-half rule (b01) and a destination-shortest-path rule (b03): both regressed. Only the narrower right/east variant (b02) helps.

## Policy observations (input for dreaming)
- v1 pruned hard after step 1: 3 branches closed, 7 attempts instead of 16. Two of the closed branches had specific repair suggestions (b01: keep only the right-half rule).
- With one branch left, steps 2-4 plateaued within noise (1.2385 / 1.2292 / 1.2396 / 1.2346). There was no exploration capacity left after the prune.
- The root (r01-b04-a04) re-measured 1.5% off its recorded value; the round-2 regressions are real.

## Ops
- Root re-measure before the round: no drift (<0.4% vs baseline).
- v1 replay: r01 V=1.1882, r02 V=1.2221, mean 1.2051.
