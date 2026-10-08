# r01-b01-a04 result: 1.2042 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 and max_abs 0.0217-0.0243 are the same as the parent, so accumulating x^2 in fp32 DST
via the ELWMUL dest-MAC is as accurate as the packer's fp32 L1 accumulation.
Per shape (parent r01-b01-a03 in brackets): h3584 14.15 µs (14.51), h4096 15.35 (15.52), h6144 19.32 (19.47),
h7168 21.49 (21.80). Score 1.2042 vs 1.1868, +1.5%. Every shape improved (-0.8 to -2.5%), but only h3584 and
h7168 are clearly outside the ±1% noise band. I expected -1 to -1.5 µs per shape (score ~1.25) and got -0.15 to -0.36 µs.

## Why (profiler evidence)
Per-core zones, all 4 devices, 5 measured calls around call idx 12 (h3584) and 51 (h7168). Times are µs from each core's
NCRISC kernel start, median [min..max] (script /tmp/r01b01a04_percore.py and inline aggregation):

| | a03 h3584 | a04 h3584 | a03 h7168 | a04 h7168 |
|---|---|---|---|---|
| PRE tail = W_PUSH end - R_INPUT end | 1.57 [1.08..2.16] | 1.28 [0.97..1.63] | 2.11 [1.34..3.62] | 1.96 [1.25..3.49] |
| W_PUSH end | 4.22 | 4.20 | 7.26 | 7.05 |
| AG wait end | 7.74 | 7.67 | 11.65 | 11.40 |
| TRISC end | 11.68 | 11.63 | 17.55 | 17.26 |
| drain end | 13.03 | 12.82 | 19.97 | 19.73 |

- The PRE tail shrank only ~0.15-0.3 µs. So the per-tile L1-acc fp32 pack was **not** PRE's limiter. With half-sync
  dest, the packer's work on block b already overlapped math on block b+1. Removing it saves the packer time on the last
  block and some sync overhead, but not the per-tile rate. That rate is set by unpack (2 x bf16 tiles per mul_tiles) or by
  HiFi4 ELWMUL math (4 fidelity phases).
- The tail still grows with width (h3584 ~1.3, h7168 ~2.0 µs). That fits compute lagging behind the deep trid read
  (data lands at DRAM rate, compute drains it more slowly), not a fixed per-row cost.
- The everything-shifts-earlier gain is the ~0.2-0.3 µs PRE saving propagated through AG, POST and drain. It does not
  change the structure: AG ~4.4 µs after the stick push, post-AG ~5.9 µs at h7168, drain tail ~2.5 µs.
- BRISC spends the first ~3.7 µs (h7168) / 2.4 µs (h3584) issuing the 2 x num_tile_cols gamma reads before it reaches
  W_PUSH. That is not on the critical path yet (W_PUSH ends 7 µs), but it would be if PRE got ~3 µs faster.

## Classification
win (small: +1.5% geomean, all shapes in the right direction, 2 of 4 clearly above noise). The DST-accumulation is
correct and free, so keep it, but the premise "PRE is pack-bound" was wrong: PRE is unpack/math (HiFi4) bound.

## What a child of this node should try next
1. **Measure before cutting PRE further.** Add a compute-side zone (DeviceZoneScopedN in the PRE loop, e.g. "C_PRE")
   so PRE start/end per tile row is visible directly instead of inferring it from W_PUSH - R_INPUT.
2. **Lower the fidelity of the x*x only.** Call the LLK directly with MathFidelity::HiFi2 (or LoFi) for the PRE mul
   (llk_math_eltwise_binary_init / llk_math_eltwise_binary with an explicit fidelity template) while POST keeps HiFi4.
   With bf16 inputs, HiFi2 drops only the lowest srcB mantissa bit (~2^-8 relative per product, mostly a near-uniform
   per-row scale bias), which should still pass pcc 0.99999 / max_abs 0.05. That halves PRE math if it is math-bound.
   If it is unpack-bound, use `mul_tiles` with srcA==srcB reuse (unpack each tile once) or square via SFPU after a
   single copy_tile.
3. The bigger structural levers are unchanged and orthogonal: column split onto this lineage (r01-b03-a02/a03), and
   the drain tail (~2.5 µs after TRISC end at h7168): split output writes across BRISC/NoC0 and NCRISC/NoC1 (NCRISC is
   idle after ~6 µs).
