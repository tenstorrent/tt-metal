# r02-b02-a03: one-matmul row-0 stat: C = ones * S^T on the DST-accumulated S = sum x*x replaces reduce<SUM,ROW> + transpose; no BRISC gather, writer back to the two 64 B face-row stick writes

## Motivation
The AG start is gated by the slowest worker's stat stick. After the last input tile lands there is a fixed ~0.9-1.0 µs
chain on every shape (r02-b02-a02 tl.py table, grandparent r02-b02-a01): pack S -> reduce<SUM,REDUCE_ROW> (init, unpack,
GAPOOL, pack, push) -> transpose (init, unpack, math, pack, push). r01-b04-a04 showed it does not scale with width.
Parent r02-b02-a02 removed both stages (x*x^T matmul, stat on the diagonal): the stat reached BRISC 0.35-0.6 µs
earlier on every shape. But the BRISC diagonal gather (32 strided volatile L1 loads + 32 stores + fences) raised
W_PUSH from 0.48 to 0.84 µs, so F_COLLECT moved only 0.03-0.2 µs and the node was neutral (1.2292 vs 1.2385).
The parent's reflection #1 asks for exactly this repair: get a row-0 stat out of the FPU with ONE op.

## Mechanism
Compute kernel only (`kernels/compute/dit_rmsnorm_fused_compute.cpp`); writer and factory revert to r02-b02-a01.
- New constexpr `mm_row_stat` = packed AG && resident (!streaming_low_l1) && !per_head_norm (the campaign config).
- PRE keeps r02-b02-a01's DST-accumulated ELWMUL S = sum_k x_k*x_k and its single pack into pre_intermediate_cb.
- Then instead of reduce + transpose: `matmul_init(reduce_scalar_sum_cb, pre_intermediate_cb, transpose=1)` and
  `matmul_tiles(sum_scalar, S)` -> C = A * S^T, where A is the existing SUM reduce-scalar tile (fp32, 1.0 in row 0 of
  every face, zeros elsewhere). C[0][i] = sum_j A[0][j] S[i][j] = sum_j S[i][j]: row 0 holds the 32 per-row sums in
  exactly the layout the transposed stick had (face_00 row 0 = rows 0-15, face_01 row 0 = rows 16-31). Pack C into
  stats_transposed_local_cb and push. The transpose stage is skipped under mm_row_stat.
- The worker writer sends its usual two 64 B face-row writes (no gather). No CB, CT-arg or protocol change.
  (The factory/writer revert to the grandparent removes the parent's stick_from_diag CT arg: host rebuild.)

## Why this is not a repeat
- r02-b02-a02 (parent): x*x^T diagonal + BRISC gather. Here the row layout comes out of the FPU, so BRISC does no
  L1 gather. The extra FPU stage is one matmul of one tile vs the grandparent's two stages (reduce + transpose).
- r01-b01-a04 / r01-b04-a04 changed only how S is accumulated; the reduce + transpose chain was untouched.

## Expected effect and risk
Stat ready ~0.2-0.3 µs earlier than r02-b02-a01 (one stage of two removed) with W_PUSH back at ~0.48 µs, so F_COLLECT,
AG, POST and the drain shift ~0.2-0.3 µs earlier: ~1-2% per shape, more on the narrow ones. Expected score ~1.25.
Risks: wrong operand orientation (would give S row 0 / col sums) -> large PCC failure. Precision: S goes through
tf32 SrcA like the old reduce, ones are exact, fp32 DST accumulation -> PCC ~0.9999985, max_abs ~0.022-0.024.
The gain is near the noise band; judge with tl.py (stat ready - R_INPUT end, W_PUSH duration, F_COLLECT end).
