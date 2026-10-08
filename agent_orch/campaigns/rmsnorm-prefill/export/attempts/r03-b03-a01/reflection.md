# r03-b03-a01 result: 1.3175 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 and max_abs 0.0204-0.0240 are identical to the parent: the fused fp32 MAD + the
same rsqrt body changes nothing visible. Per shape, this node vs the parent r02-b02-a03 (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 13.95 | 12.86 | -1.08 (-7.8%) |
| h4096 | 15.12 | 14.07 | -1.05 (-6.9%) |
| h6144 | 18.67 | 17.86 | -0.81 (-4.3%) |
| h7168 | 20.38 | 19.46 | -0.92 (-4.5%) |

Score 1.3175 vs 1.2396 (+6.3%). Every shape is far outside the ±1% noise band, and this is the new campaign best. I
predicted -0.5 to -1.0 µs per shape and the result is at the top of that range.

## Why (profiler evidence)
`comb.py` (the r02-b02-a04 script, copied here; output in `comb_out.txt`) reads the same C_COMB / C_POST zones on
TRISC_0 (unpack). Medians over the measured calls, all 4 chips, all 20 workers. r02-b02-a04 -> this node:

| shape | C_COMB end -> C_POST start (the math+pack combine gap) | POST unpack dur | drain end - POST end |
|---|---|---|---|
| h3584 | 1.281 -> **0.384** | 1.77 -> 1.77 | 1.02 -> 1.08 |
| h4096 | 1.273 -> **0.379** | 2.03 -> 2.03 | 1.29 -> 1.26 |
| h6144 | 1.280 -> **0.379** | 3.07 -> 3.07 | 1.91 -> 1.87 |
| h7168 | 1.274 -> **0.379** | 3.58 -> 3.58 | 1.86 -> 1.82 |

- **The combine gap shrank by 0.90 µs on every shape**, as predicted. That 0.9 µs (~1200 cycles) was the three
  full-tile SFPU passes: rsqrt over 32 iterations (the non-approx SQRT_23 body) dominated, plus mul 1/H and add eps.
  The fused add_rsqrt on row 0 does 4 iterations.
- POST and the drain tail after POST are unchanged, so the whole post-AG section just shifts ~0.9 µs earlier. The
  kernel end moved 0.8-1.1 µs, which confirms the drain is throughput-bound from its start: starting it earlier ends
  it earlier.
- **What is left in the gap (0.38 µs):** 2 ELWADDs (already unpacked within C_COMB's 0.14 µs), `transpose_dest<fp32>`
  (a 32-bit transpose via hi/lo 16-bit halves, ~30 MOV ops + cfg RMWs), the 1-tile fp32 pack, the reduce_result CB
  handoff, and POST's reconfig + mul_bcast_cols_init.

## Classification
win (+6.3% geomean, all shapes far outside noise; new campaign best). It implements r02-b02-a04's measured
bottleneck #1.

## What a child of this node should try next
1. **Cut the remaining 0.38 µs combine gap.**
   - Drop `transpose_dest<fp32>`. Accumulate the 4 gathered row-0 tiles with matmuls `I * G_d^T`
     (`matmul_init(identity_cb, gathered_cb, transpose=1)`, 4 accumulating matmul_tiles into DST 0). That gives the
     col-0 sum directly (C[r][0] = sum_k I[r][k] G[0][k] = s[r]). Then the add_rsqrt has to run with
     `VectorMode::C` (faces 0+2, 8 iterations each = 16) instead of R. It needs an identity tile in a CB (writer
     fills it at start, like the reduce scalars).
   - Alternatively, keep everything and hoist POST's `reconfig_data_format` / `mul_bcast_cols_init` /
     `pack_reconfig` ahead of the `cb_reduce_result.wait_front`. Unpack can init while math/pack finish the combine.
   - Measure with this node's C_COMB / C_POST zones (keep them) and `comb.py`.
2. **The same trick for PRE/POST elsewhere:** look for any other full-tile SFPU op whose live data is a row or a
   column. The x*gamma pre-pass and POST are FPU, so this is likely the only one.
3. **The bigger remaining costs are unchanged:** the drain tail ~1.1-1.9 µs after POST ends (r02-b02-a01's per-core
   NoC0 share tuning), the AG section ~3 µs (F_COLLECT -> go), and the stick DRAM read after go (~0.67 µs).
