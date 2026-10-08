# r03-b02-a01: post-AG stat combine on row 0 only: one fused SFPU add_rsqrt (x*1/H + eps, rsqrt) over the 4 SFPU iterations that hold tile row 0, run BEFORE transpose_dest, instead of three full-tile SFPU passes after it; POST's unpack reconfig/init hoisted ahead of the reduce_result wait

## Motivation
r02-b02-a04 (sibling of this round's root, same compute) put TRISC_0 zones around the post-AG combine and found the
largest fixed cost after the AG: the unpacker finishes the combine's unpacks 0.13 µs after the gathered sticks land, then
idles **1.27-1.28 µs on every shape** before the first POST tile. That gap is the math + pack half of the combine:

    ELWADD t0+t1, ELWADD += t2+t3       (FPU, full tile)
    transpose_dest<fp32>                (row 0 -> col 0)
    mul_unary_tile(1/H)                 SFPU, VectorMode::RC: 4 faces x 8 iterations = 32 SFPU iterations
    add_unary_tile(eps)                 SFPU, 32 iterations
    rsqrt_tile                          SFPU, 32 iterations of the ~25-instruction rsqrt body (non-approx, 1 NR step)
    pack fp32 tile -> reduce_result_cb, CB handoff, POST unpack reconfig + mul_bcast_cols_init

The SFPU part alone is ~96 iterations, roughly 1000+ cycles (~0.75 µs at 1.35 GHz), of which only the lanes holding
the 32 per-token stats matter. The stat lives in ONE row (row 0 before the transpose, col 0 after). It sits directly on the
critical path: AG end -> combine -> POST -> drain end.

## Mechanism
Compute kernel only (`dit_rmsnorm_fused_compute.cpp`, packed-AG + `stats_tiles_cols > 1` branch, BH):
1. After the two ELWADDs (sum of the 4 gathered row-0 tiles in DST), run ONE fused SFPU op on row 0:
   `add_rsqrt_tile<false, VectorMode::R, /*ITERATIONS=*/2, false, recip_h_full_bits>(0, eps_bits)`
   (`api/compute/experimental/add_rsqrt.h`, `calculate_add_rsqrt`: y = rsqrt(x * INPUT_SCALE + eps), same
   `_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST=false>` as rsqrt_tile). On BH one SFPLOAD covers 4 rows x 8
   columns of one parity (ckernel_sfpu_triangle_solve.h / rope.h lane-map comments), so 2 iterations = rows 0-3, all 16
   columns of a face. VectorMode::R does faces 0 and 1, and the face step (`SETRWC CR_D`) is relative to the face base,
   so it doesn't matter that only 2 of 8 iterations ran. Total: 4 SFPU iterations instead of 96.
2. Then transpose_dest<fp32> moves the finished 1/rms from row 0 to col 0 (pure data movement, exact for fp32), pack.
   Rows 1-31 hold garbage before and after, exactly as before (bcast_cols reads col 0 only).
3. POST (prescale path): issue `reconfig_data_format(intermediate, reduce_result)` + `mul_bcast_cols_init` BEFORE
   `cb_reduce_result.wait_front(1)`, so the unpacker sets up POST while math/pack run the combine. Math and pack
   execute in program order regardless (wait_front is unpack-side).
4. Keep r02-b02-a04's TRISC zones (C_COMB around the combine, C_POST right after the reduce_result wait) to measure the gap.
Non-BH builds keep the old code path (`#if defined(ARCH_BLACKHOLE)`).

## Why this is not a repeat
- r02-b02-a04 only *measured* the combine (its mechanism was the forwarder multicast go). Its reflection proposes
  exactly this as next step #1 ("run mul/add/rsqrt only on the faces that hold the stat ... fold *1/H"). Nobody has
  changed the post-AG combine since the root.
- r02-b02-a02/a03 attacked the PRE-side stat tail (before the AG); this is the post-AG side.
- Not a per-tile POST throughput tweak (fidelity etc.); it removes fixed per-row work.

## Expected effect and risk
- Expected: C_COMB-end -> C_POST-start gap 1.28 -> ~0.5-0.6 µs; every shape -0.5..-0.7 µs (it is a fixed cost), so
  ~+3-4% geomean (larger relative gain on h3584/h4096).
- Accuracy: same rsqrt body; x*s+eps is one fp32 MAD instead of mul then add (one rounding fewer). PCC/max_abs should be
  unchanged (~0.9999985 / ~0.024).
- Risks: (a) wrong lane coverage (if 2 iterations did not cover row 0 fully) -> some tokens get un-normalized stats ->
  PCC failure, easy to see; (b) the experimental add_rsqrt header fails to JIT-compile -> fix and retry before device run;
  (c) unpack-side init before the wait races with the transpose_dest dummy srcB valid -> hang/garbage (would show as
  hang or accuracy_fail); fallback is to drop step 3.
