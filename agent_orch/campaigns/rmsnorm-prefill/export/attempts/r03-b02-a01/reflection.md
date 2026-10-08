# r03-b02-a01 result: 1.3231 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 everywhere and max_abs is 0.0204-0.0240, identical to the root r02-b02-a03, so the
row-0-only fused `add_rsqrt` (one fp32 MAD sum*1/H+eps, then the same rsqrt body) is as accurate as the three full-tile
passes it replaces. It also confirms that 2 SFPU iterations per face (VectorMode::R) cover all 32 columns of tile row 0 on BH.
Per shape against the root (µs, chip mean):

| shape | root r02-b02-a03 | this node | change |
|---|---|---|---|
| h3584 | 13.95 | 12.73 | -1.22 (-8.8%) |
| h4096 | 15.12 | 14.13 | -0.99 (-6.6%) |
| h6144 | 18.67 | 17.72 | -0.95 (-5.1%) |
| h7168 | 20.38 | 19.42 | -0.96 (-4.7%) |

Score 1.3231 vs 1.2396 (+6.7%). Every shape is far outside the ±1% noise band. This is the new campaign best. I expected
-0.5..-0.7 µs per shape and got ~-1 µs.

## Why (profiler evidence)
`comb.py` (in this dir; `python3 comb.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`; output for r02-b02-a04 and this node
in `comb_out.txt`) reads the TRISC_0 (unpack) zones that r02-b02-a04 introduced. It reports medians over measured calls
and all 4 chips, per worker. The key number is `post_s-comb_e` on TRISC_0: the time from the end of the combine's
unpacks to the first POST unpack, i.e. the combine's math + pack + CB handoff.

| shape | r02-b02-a04 (same combine as root) | this node |
|---|---|---|
| h3584 | 1.281 | 0.384 |
| h4096 | 1.273 | 0.379 |
| h6144 | 1.280 | 0.379 |
| h7168 | 1.274 | 0.380 |

- **The fixed post-AG gap shrank by 0.9 µs on every shape**, as a fixed cost should. So ~0.9 µs of the 1.28 was the 96
  full-tile SFPU iterations (mul_unary + add_unary + the ~25-instruction non-approx rsqrt body) plus their inits.
  4 iterations of the fused op cost almost nothing. My cycle estimate (~0.75 µs) was low; the rsqrt body with v_if
  predication is costlier per iteration than I assumed.
- The gain lands 1:1 in the kernel end. POST unpack throughput is unchanged (TRISC_0 POST 1.71 / 1.97 / 3.00 / 3.52 µs
  vs 1.77 / 2.03 / 3.07 / 3.58), and the drain tail after pack's POST end is unchanged (1.07 / 1.27 / 1.90 / 1.80 µs vs
  1.02 / 1.29 / 1.91 / 1.86). The whole post-AG chain just starts 0.9 µs earlier. The extra ~0.1-0.3 µs on the chip
  mean beyond the 0.9 is probably the shorter op tightening the next call's start skew (r02-b02-a01 saw the same coupling).
  This is not separately measured.
- The hoisted POST unpack reconfig/init (before the 1/rms wait) is part of the 0.38 µs number, but it can't be separated
  from the SFPU change. It is at most a few tens of ns.
- Caveat for readers: the TRISC_2 `C_POST` zone now starts after its own pack reconfig. The pack thread doesn't wait on
  reduce_result_cb, so its "post_s-comb_e" / "post dur" are not comparable with r02-b02-a04. Use TRISC_0 only.

## Classification
win (+6.7% geomean over the round root, every shape -0.95..-1.22 µs, far outside noise). It removes ~0.9 µs of fixed
post-AG critical-path work that r02-b02-a04's zones located.

## What a child of this node should try next
1. **The remaining 0.38 µs combine gap.** What's left: 2 full-tile ELWADDs, transpose_dest<fp32> (cfg RMWs + MOVs),
   a full 4 KB fp32 pack to reduce_result_cb, and the CB handoff to unpack. Options:
   (a) replace ELWADD x2 + transpose_dest with 4 accumulating matmuls C += I * G_d^T (identity in in0, the gathered
   tile in in1 with transpose=1). The output is the transposed sum in col 0 directly, but it needs an identity tile CB
   (writer-filled, bf16), and HiFi4 matmul x4 may cost more than it saves. Measure with C_COMB/C_POST on TRISC_0.
   (b) a cheaper pack: only faces 0/2 carry col 0. That would need a partial-face pack config, which is risky.
   Expect at most ~0.2 µs. Diminishing returns.
2. **Same trick on the PRE side (pre-AG critical path).** The PRE row-stat path still does a full-tile DST pack of
   S -> L1 -> unpack -> matmul ones*S^T -> full-tile pack (r02-b02-a03: that handoff is the PRE tail). Any full-tile
   SFPU/FPU work there that could be restricted to row 0 is worth checking with a TRISC zone around PRE.
3. **Larger levers on this lineage, still unexplored:**
   - The column split (r01-b03-a02/a03, k=4 / 80 workers, one kernel group, bank rotation) was never ported onto the
     r01-b04/r02-b02 lineage. POST (~64 ns/tile unpack) and the drain are now the biggest post-AG costs, and both scale
     with tiles per core.
   - The drain tail after POST (1.1-1.9 µs) is unchanged.
   - Per-core NoC0 share tuning (r02-b02-a01 #1).
4. Keep the C_COMB / C_POST TRISC_0 zones. They are cheap and they made this node's diagnosis possible.
