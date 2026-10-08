# r03-b03-a01: post-AG stat chain on row 0 only: one fused SFPU add_rsqrt (x*1/H + eps, rsqrt) with VectorMode::R over 2 iterations/face BEFORE the fp32 transpose_dest, instead of three full-tile SFPU passes after it

## Motivation
r02-b02-a04 put TRISC zones on the combine. Its finding: after the gathered sticks land, the unpacker spends
0.13 µs unpacking the combine inputs and then idles **1.27-1.28 µs on every shape** before POST starts. That gap is
the math + pack half of the combine: 2 ELWADDs, `transpose_dest<fp32>`, `mul_unary_tile` (1/H), `add_unary_tile`
(eps), `rsqrt_tile`, pack to reduce_result_cb, and the CB handoff. It is a fixed cost on the critical path between
"all stats in L1" and the first POST tile, so the drain (the kernel end) starts that much later.

The three SFPU ops each run with `VectorMode::RC`, i.e. 4 faces x 8 iterations = 32 SFPU iterations each:
- `rsqrt_tile` non-approx is the SQRT_23-bit body (~25 SFPU instructions plus v_if/v_else, ~30 cycles/iteration),
  so about 1000 cycles.
- mul + add add another ~2 x 32 x ~6 cycles.
- That totals ~1300-1400 cycles, about 1.0 µs at 1.35 GHz. That is most of the measured 1.28 µs.

Only 32 values matter: the per-token stat. In the code today they sit in ROW 0 after the adds, and the code transposes
them to col 0 first, so the SFPU then has to sweep all 4 faces.

## Mechanism
In `dit_rmsnorm_fused_compute.cpp`, packed-AG combine branch (stats_tiles_cols > 1, packed_ag_enabled):
1. Keep the 2 ELWADDs (row-0 sum of the 4 gathered tiles in DST, fp32).
2. NEW: while the stat is still in row 0, run ONE fused SFPU op,
   `add_rsqrt_tile<false, VectorMode::R, 2, false, recip_h_full_bits>(0, eps_bits)`
   (api/compute/experimental/add_rsqrt.h: y = rsqrt(x * 1/H + eps), fp32, same `_calculate_sqrt_body_` and
   APPROX as `rsqrt_tile`). VectorMode::R runs faces 0 and 1 only. ITERATIONS=2 covers dest rows 0-3, both even and odd
   columns (one SFPLOAD = 4 rows x 8 cols, odd columns at +2; see ckernel_sfpu_triangle_solve.h). That is
   4 SFPU iterations instead of 96.
3. Then `transpose_dest<fp32>` moves row 0 to col 0 as before. It STALLWAITs on SFPU, so the ordering is safe. Its
   init now runs after the SFPU init.
4. Pack + POST unchanged. The other rows/cols of the tile are don't-care (bcast_cols reads col 0 only), as before.
5. Guarded by `#ifdef ARCH_BLACKHOLE` (add_rsqrt is BH-only). Other archs keep the old chain.
6. Add the r02-b02-a04 zones C_COMB / C_POST (DeviceZoneScopedN) so the combine gap is measured directly and is
   comparable to r02-b02-a04's `comb.py`.

Files: `device/kernels/compute/dit_rmsnorm_fused_compute.cpp` only (kernel-only, JIT, no host rebuild).

## Why this is not a repeat
- r02-b02-a04 measured the gap and proposed cutting it ("run mul/add/rsqrt only on the faces that hold the stat",
  "fold 1/H"). Nobody has implemented it.
- r02-b02-a02/a03 attacked the PRE stat chain (reduce/transpose) before the AG, not the post-AG combine.
- This differs from VectorMode::C after the transpose (16 iterations): doing the SFPU work before the transpose on
  row 0 needs only 4 iterations, and it fuses 3 SFPU calls and 2 inits into 1.

## Expected effect and risk
- Expected: the combine gap shrinks by ~0.7-1.0 µs. The first POST tile, the drain start and (because the drain is
  throughput-bound from its start) the drain end all move earlier by about that much on every shape. That is
  -0.5 to -1.0 µs/shape, roughly +3-6% score.
  If the drain is bound by something else, the gain would show up in C_POST start but not in the kernel end.
- Accuracy: rsqrt is the same body. x*s+eps is one fp32 MAD instead of a mul then an add (≤1 ulp difference).
  PCC/max_abs should be unchanged.
- Risk: the VectorMode::R/ITERATIONS lane mapping is wrong, so part of row 0 is not processed. That would be a
  gross accuracy failure (columns of tokens without rsqrt) and would show as accuracy_fail in the eval.
- Risk: transpose_dest after SFPU state. If it is wrong it would also show as accuracy_fail or a hang.
