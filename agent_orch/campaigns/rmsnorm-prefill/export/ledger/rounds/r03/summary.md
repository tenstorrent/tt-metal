# Round r03 summary (policy v2, W=4, R=3, root r02-b02-a03 = 1.2396)

12 attempts in 3 steps. No branch closed: every anchor stayed within 6.8% of the leader. All committed, verified and valid.

## Best
r03-b02-a02, score 1.3333: 12.59 / 14.05 / 17.55 / 19.32 µs (1.350 / 1.299 / 1.338 / 1.347x). PCC unchanged.
Tied within noise: r03-b04-a03 (1.3318, best h7168 at 18.72 µs, 1.390x) and r03-b01-a03 (1.3255, best h3584 at 12.54 µs, 1.355x).

## Grid (score)
| branch | a01 | a02 | a03 |
|---|---|---|---|
| b01 | 1.313 | 1.322 | 1.326 |
| b02 | 1.323 | 1.333 | 1.300 |
| b03 | 1.318 | 1.312 | 1.302 |
| b04 | 1.313 | 1.244 | 1.332 |

## What worked
- Post-AG stat finalize on row 0 only: fused SFPU add_rsqrt before transpose_dest. All 4 branches found it in step 1 (+6% over root).
- Stats scratch in L1 instead of DRAM (b01-a02, b02-a02).
- Gamma streamed to compute in 8-page chunks so the x*gamma pre-pass starts at PRE end (b04-a03, best on h7168).
- Stick push decoupled from the gamma read barrier (b01-a03, best on h3584).

## What failed and why
- More output-drain parallelism: second command buffer (b02-a03) and round-robin VCs (b03-a03) both regressed. Reflections say to measure the drain tile cost before trying another drain mechanism.
- Two-wave AG pipeline on 20 cores (b04-a02): regressed; reverted by b04-a03.
- PRE stat straight out of DST (b03-a02): neutral.

## Policy observations (input for dreaming)
- All four step-1 branches converged on the same mechanism (they all followed the reflection of r02-b02-a04). Width bought robustness, not diversity, in step 1.
- Width paid off in steps 2-3: three different winning mechanisms on three branches, each best on a different shape.
- The 6.8% gap rule never fired: all branches stayed close after a strong common first step.
- 3rd attempts: one repair win (b04: 1.244 -> 1.332), one small gain (b01), two regressions (b02, b03).
- Round 4 should stack the complementary pieces (L1 scratch + gamma streaming + stick-push decoupling) on r03-b02-a02.

## Ops
- Root re-measure: no drift. v2 replay: r01 1.1942, r02 1.2296, r03 1.3133, mean 1.2457.
