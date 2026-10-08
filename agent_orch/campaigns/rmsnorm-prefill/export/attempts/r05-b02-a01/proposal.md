# r05-b02-a01: PRE row stat as the diagonal of a DST-accumulated x·xᵀ matmul (HiFi4), moved into row 0 inside DST by an SFPU identity mask + SFPU column sum, replacing the ELWMUL x·x + S pack → unpack → ones·Sᵀ matmul round trip

## Motivation
The AG start is gated by the slowest worker's stat. At HiFi4 (the campaign rule now forbids the HiFi2 PRE of
r04-b01-a02/a03), the time from the last input tile landing to the stat being ready is ~0.8 µs median and ~1.1 µs on the
slowest core on every shape (r04-b01-a03's `pre.py` on r04-b04-a02: 0.67/0.79/0.83/0.82 med, 0.85/1.07/1.15/1.13 max).
HiFi2 showed that this tail maps 1:1 onto the kernel end (-0.4..-0.6 µs on every shape), so it is the most valuable
pre-AG lever left. Two parts:
- the per-tile x·x ELWMUL (HiFi4, 4 fidelity phases, DST-accumulated) lags the read near the end of the row;
- the fixed hop after the last mul: pack S (fp32) → L1 → unpack S + ones → `matmul(ones, Sᵀ)` → pack the stat.

The only measured alternative is r02-b02-a02: the stat as the diagonal of a DST-accumulated `x_k · x_kᵀ` matmul,
packed straight out of DST (no S round trip). Its stat was ready ~0.35-0.49 µs earlier at the slowest core than
r02-b02-a03's ELWMUL + ones·Sᵀ path (same lineage, same HiFi4; derived from both nodes' `tl.py` push-end max minus push
duration: 4.01 vs 4.45, 4.73 vs 5.08, 6.66 vs 7.15, 7.29 vs 7.67 µs), with the same R_INPUT end. That node lost it
again because BRISC gathered the 32 strided diagonal words (+0.36 µs W_PUSH, cache-missing volatile loads).
r02-b02-a03 #1(c) suggested doing the diagonal → row 0 move inside DST instead; nobody tried it.

## Mechanism
Compute kernel (`dit_rmsnorm_fused_compute.cpp`), resident whole-row packed-AG RMS PRE only (the `mm_row_stat` path the
campaign shapes take), Blackhole, no fused RoPE:
1. Before the first input block: `copy_tile` an identity tile into DST[1] (bf16 → fp32, exact), then
   `matmul_init(input, input, transpose=1)`. These happen while compute is waiting for the first input block, so they
   are off the critical path.
2. Per input tile: `matmul_tiles(input, input, k, k, 0)` accumulates C = Σ_k x_k·x_kᵀ in fp32 DST[0] at the kernel's
   MathFidelity (HiFi4). C[i][i] = Σ_j x[i][j]², the per-row sum of squares (bf16 products exact at HiFi4).
3. After the last tile: SFPU `mul_binary_tile(0, 1, 0)` zeroes everything but the diagonal (fp32, exact), then
   `sfpu_reduce<SUM, Float32, REDUCE_COL>(0)` sums each column into row 0, so row 0 col i = C[i][i]: the same row-0
   stick layout the writer already sends (two 64 B face rows). One pack to stats_transposed_local_cb.
   No S pack, no L1 round trip, no second unpack, no ones·Sᵀ matmul. The stat stays fp32 end to end (today S is
   truncated to tf32 in SrcA before the ones matmul).

Worker writer (`dit_rmsnorm_fused_worker_writer.cpp`): the identity tile is built in the rope transformation-matrix
CB (c_11: always allocated, bf16, 1 tile, unused when RoPE is not fused): NoC zero-fill + 32 diagonal 1.0 stores,
right after the reduce scalars. RMS only (num_stats == 1), not when RoPE is fused.

Kernel-only change, no host rebuild. Fallback paths (WH, fused RoPE, streaming, per-head) keep today's code.

## Why this is not a repeat
- r02-b02-a02: same matmul, but the diagonal was gathered on BRISC (+0.36 µs on the push). Here the gather never
  leaves DST; the writer is unchanged except for building the identity at start.
- r02-b02-a03 (ones·Sᵀ, today's code) keeps the S round trip.
- r03-b03-a02: removed the S round trip with fp32 `transpose_dest` + SFPU column sum on S: neutral, probably because
  the 32-bit transpose_dest is expensive. Here no transpose is needed: C is symmetric and the masked diagonal's
  column sums *are* the diagonal, so only the cheap column sum (4 SFPU passes) plus one fp32 SFPU multiply remain.
- r04-b01-a02 changed fidelity (now forbidden). This keeps HiFi4 for every FPU op.

## Expected effect and risk
- Stat ready (W_PUSH start) earlier relative to the last input tile by ~0.15-0.35 µs at the slowest core: the
  r02-b02-a02 gain (~0.35-0.5 µs) minus the SFPU mask + column sum (~0.15-0.25 µs). Since the AG is gated by the local
  pushes (r04-b01-a03), the kernel end should move by about that much on all shapes: ~+1-2.5% score (~1.41-1.43).
- If the matmul per tile were slower than the ELWMUL, PRE would lag the read more and the gain would shrink or turn
  negative; r02-b02-a02's data says it isn't.
- Accuracy: same products, different summation order; r02-b02-a02 measured max_abs 0.024-0.025 with this matmul
  (gate 0.05), PCC unchanged. A wrong mask/orientation would collapse PCC (accuracy_fail).
- JIT compile errors possible (SFPU binary / reduce APIs in this kernel); no hang risk (CB counts: one extra tile
  pushed by the writer, waited/popped once by compute).
- Check with `pre.py` (r04-b01-a02/r04-b01-a03): "stat ready - input end" med/max and push end max vs the parent.
