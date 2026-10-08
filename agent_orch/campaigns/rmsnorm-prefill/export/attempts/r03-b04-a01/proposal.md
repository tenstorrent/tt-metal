# r03-b04-a01: post-AG stat finalize on row 0 only: *1/H, +eps and rsqrt run as 2-iteration SFPU passes over faces 0/1 BEFORE transpose_dest, instead of full-tile 32-iteration passes after it

## Motivation
r02-b02-a04 (sibling lineage, same compute as this root r02-b02-a03) added TRISC zones and found the largest fixed
post-AG cost: after the gathered sticks land, the unpacker finishes its C_COMB part in 0.13 µs and then idles
**1.27-1.28 µs** (every shape) until C_POST can start. That idle is the math + pack half of the combine:
`add_tiles` x2 (ELWADD, row-0 tiles) -> `transpose_dest<fp32>` (row 0 -> col 0) -> `mul_unary_tile(1/H)` ->
`add_unary_tile(eps)` -> `rsqrt_tile` -> pack. It sits on the critical path: the drain is throughput-bound from the
first output tile, and no output tile exists until the combine ends.

The three SFPU ops run on the WHOLE 32x32 fp32 tile (VectorMode::RC, 4 faces x 8 iterations = 32 SFPU iterations
each). `rsqrt` is the fp32 SQRT_23-bit reciprocal body (~30+ SFPU instructions per iteration, see
ckernel_sfpu_sqrt.h `_calculate_sqrt_body_`), so it alone is ~1000 cycles (~0.75 µs). mul + add are ~8
instructions/iteration each, another ~500 cycles. That is most of the 1.27 µs. But only 32 values matter: the
per-row stats. After the sum they sit in row 0 (face 0 row 0 = rows 0-15, face 1 row 0 = rows 16-31), and POST's
`mul_tiles_bcast_cols` only reads column 0 after the transpose.

## Mechanism
Compute kernel only (`device/kernels/compute/dit_rmsnorm_fused_compute.cpp`), packed-AG combine branch:
- Reorder to add -> SFPU finalize -> transpose. The finalize is elementwise, so applying it before the transpose
  gives the same col-0 values after it.
- Run the finalize only on row 0, through the same LLK functors the tile APIs use, but called with
  `VectorMode::R` (faces 0 and 1 only) and `ITERATIONS = 2`. One SFPLOAD covers 4 rows x 8 columns of a face, with the
  odd columns at address +2 (ckernel_sfpu_triangle_solve.h / ckernel_sfpu_reshuffle_rows.h), so iterations 0-1 cover
  rows 0-3 of the face, which includes all of row 0. That is 4 SFPU iterations per op instead of 32.
  `calculate_binop_with_scalar<APPROX, MUL/ADD, 2, fp32>` and `calculate_rsqrt<APPROX, 2, fp32, false>` via
  `SFPU_UNARY_CALL`, same inits as before.
- `transpose_dest<true>` then moves the finished row 0 into column 0. Rows 4-31 hold unfinalized sums/garbage that
  POST never reads (col 0 after the transpose = row 0 before it).
- Add the r02-b02-a04 instrumentation zones `C_COMB` / `C_POST` (unpack-side timing; only zones), so this node's
  combine gap can be compared directly with a04's 1.27 µs.
No host change, no CB change, no protocol change (kernel JIT only).

## Why this is not a repeat
- r02-b02-a04 measured the combine but changed only the forwarder/go path. Its reflection #1 lists "run
  mul/add/rsqrt only on the faces that hold the stat" as the cheapest option. Nobody has touched the combine math.
- r02-b02-a02/a03 and r01-b0x-a04 worked on the PRE side (before the AG). This is the post-AG side.
- The drain/NoC nodes (r01-b02-a04, r01-b03-a04, r02-b0x-a01) changed bytes on links, which this does not.

## Expected effect and risk
- The SFPU work in the combine drops ~8x, so the unpack gap goes from ~1.27 µs to ~0.4-0.6 µs. POST, the first
  output tile and the drain all start ~0.6-0.8 µs earlier, so the kernel ends ~0.6-0.8 µs earlier on every shape.
  That is ~4-5% at h3584/h4096 and ~3% at h6144/h7168, a score of roughly 1.28-1.29.
- Accuracy: the same functions run on the same fp32 values, so the output should be bit-identical (PCC 0.9999985,
  max_abs 0.0204-0.0240 as in the root).
- Risk: if the SFPU lane layout were different, some rows would skip the finalize and come out wildly wrong
  (accuracy_fail, a clear signal). JIT compile errors would show up as jit_compile_error.
- Judge with comb.py (r02-b02-a04's script): "TRISC_0 post_s-comb_e" should fall from ~1.27 µs.
