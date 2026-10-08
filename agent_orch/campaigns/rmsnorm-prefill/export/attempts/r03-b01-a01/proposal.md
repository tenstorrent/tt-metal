# r03-b01-a01: post-AG stat combine on the stat row only (SFPU VectorMode::R, 2 iterations/face) before transpose_dest

## Motivation
r02-b02-a04's TRISC_0 zones measured the post-AG fixed cost: after the gathered sticks land, the unpacker idles
**1.27-1.28 µs on every shape** before C_POST starts. That gap is the math+pack half of the stat combine
(`dit_rmsnorm_fused_compute.cpp`, packed-AG branch of P_NRED): 2 ELWADDs, `transpose_dest<fp32>`, then
`mul_unary_tile(1/H)`, `add_unary_tile(eps)`, `rsqrt_tile` on the whole 32x32 fp32 tile (VectorMode::RC: 4 faces x
8 SFPU iterations = 32 iterations each), pack to reduce_result_cb. It sits on the critical path between "all stats in
L1" and the first POST tile, so the whole POST + drain shift by it. That reflection ranked it #1 for a child.

Only 32 values of the tile are meaningful. Before the transpose they are row 0 (faces 0 and 1); after it, col 0.
The SFPU work is 32 iterations x 3 ops, and the rsqrt (fp32 `_calculate_sqrt_body_` with reciprocal, Newton steps)
is ~20-30 SFPU instructions per iteration. That is on the order of 1000+ math-thread cycles (~0.7-0.9 µs at 1.35 GHz)
spent on 992 garbage values.

## Mechanism
In the packed-AG `stats_tiles_cols > 1` branch only (the configuration all four campaign shapes run):
- Keep the FPU ELWADD of the ring_size gathered row-0 tiles into DST 0.
- Run `*1/H_full`, `+eps`, `rsqrt` **before** `transpose_dest`, on the row-0 stat only: call the same SFPU functions
  through `SFPU_UNARY_CALL` with `VectorMode::R` (faces 0 and 1) and `ITERATIONS = 2`. On BH one SFPU iteration covers
  4 dest rows x 8 columns (even or odd columns, see the `_generic_moe_gate_*` comment in tt-llk), so iterations 0-1
  cover rows 0-3 x all 16 columns of a face, which contains row 0. 4 iterations per op instead of 32.
- Then `transpose_dest<fp32>` moves the finished 1/rms values from row 0 to col 0 (elementwise ops commute with the
  transpose). transpose_dest stalls on WAIT_SFPU, so ordering is safe. The rest of the tile is garbage, as before;
  `mul_tiles_bcast_cols` only reads col 0.
- Same functions, same fp32 DST, same constants, so the values are bit-identical to the parent.

File: `device/kernels/compute/dit_rmsnorm_fused_compute.cpp` only. Kernel-only change (JIT, no host rebuild).

## Why this is not a repeat
- r02-b02-a04 found the cost but changed the forwarder/go release (neutral). Not in this lineage.
- r02-b02-a02/a03 changed the PRE side (pre-AG stat production). This is the POST side after the AG.
- No node has touched the combine's SFPU work or its vector mode.

## Expected effect and risk
- Expected: -0.5 to -0.9 µs on every shape (fixed cost, so relatively bigger on h3584/h4096), score ~1.27-1.29 if
  POST start shifts the drain 1:1. Less if the drain end is set by another core.
- Risk: if my SFPU row-mapping assumption is wrong, part of row 0 misses the rsqrt -> accuracy_fail, obvious in the
  eval. Hang risk is low (no CB/sync change).
