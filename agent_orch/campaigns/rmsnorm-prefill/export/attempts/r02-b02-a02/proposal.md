# r02-b02-a02: sum(x^2) per row as the diagonal of a DST-accumulated x·x^T matmul; BRISC gathers the 32 diagonal words into the stick, so the reduce + transpose leave the AG-start path

## Motivation
The AG start is gated by the slowest worker's stat stick, and that path has a fixed per-row compute tail after the
last input tile lands. Parent r02-b02-a01 profile (`reports/r02-b02-a01`, all 4 chips, measured calls, medians, µs
from the first worker kernel start; script /tmp/r02b02a02/tl.py):

| shape | R_INPUT end (min-max) | stat ready = W_PUSH start (min-max) | W_PUSH NoC part | F_COLLECT end | AG end |
|---|---|---|---|---|---|
| h3584 | 2.90-3.37 | 3.75-4.35 | ~0.48 | 4.94 | 7.95-8.30 |
| h7168 | 5.25-6.46 | 6.18-7.51 | ~0.48 | 8.05 | 11.20-11.56 |

So ~0.9-1.0 µs passes between the last input tile landing and the stat being available to BRISC, on every shape
(it does not scale with width). r01-b04-a04 showed this tail is not per-tile pack cost (DST accumulation removed the
per-tile L1-acc pack and the tail did not move): it is the fixed chain after the last `mul_tiles`:
pack S -> push pre_intermediate -> `reduce<SUM,REDUCE_ROW>` (init, unpack, GAPOOL, pack, push stats_local) ->
`transpose_init` + `transpose_tile` + pack + push stats_transposed_local. That is two extra full unpack/math/pack
round trips with re-inits and three cross-TRISC CB handoffs, all on the AG-start critical path.
r01-b04-a04's reflection #1 asked exactly for this: "attack the fixed PRE tail ... produce the row sums directly
in row 0 instead of reduce-to-col-0 then transpose".

## Mechanism
- Compute (`kernels/compute/dit_rmsnorm_fused_compute.cpp`), resident whole-row packed-AG PRE only
  (`diag_stat = packed_ag_enabled && !streaming_low_l1 && !per_head_norm`): instead of `mul_tiles(x,x)` into DST
  followed by reduce + transpose, run `matmul_init(input, input, transpose=1)` and accumulate
  `matmul_tiles(input, input, k, k, 0)` over the row in DST 0 under one acquire (matmul accumulates into DST, the
  same way the parent's ELWMUL accumulated). DST 0 then holds C = sum_k x_k x_k^T, whose diagonal
  C[i][i] = sum_k sum_j x_k[i][j]^2 is exactly the per-row sum of squares (fp32 DST, bf16 products exact at
  HiFi4, no tf32 truncation of S before a reduce). Pack C once straight into stats_transposed_local_cb and push.
  The reduce and the transpose are skipped on this path. Streaming / per-head / TP=1 keep the old code.
- Worker writer (`kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`): new trailing CT arg `stick_from_diag`.
  When set, `push_stick` reads the 32 diagonal fp32 words of the stat tile (face_00 word 17i, face_11 word
  768+17(i-16)), stores them contiguously at the tile's start (in place, after a fence), and sends the 128 B stick
  with ONE NoC write instead of two 64 B face-row writes. The stick content (32 per-row sums in row order) is
  identical to the old transposed row 0, so the forwarder and the post-AG compute are untouched.
- Factory: `stick_from_diag = use_mux && !is_layernorm && !streaming_low_l1 && !per_head_norm`, appended to the
  worker-writer CT args (host rebuild).

## Why this is not a repeat
- r01-b01-a04 / r01-b04-a04 changed how x^2 is accumulated (DST MAC instead of L1-acc pack): same reduce + transpose
  chain afterwards, and their measurements showed that chain is the remaining fixed tail. No node has removed the
  reduce or the transpose.
- r02-b01/b02/b03 and most of r01-b02/b03 worked on the output drain; this is the pre-AG side, a different part of
  the critical path, and stacks with the parent's dual-NoC drain.

## Expected effect and risk
- Stat ready ~0.3-0.6 µs earlier after the last input tile, so F_COLLECT, the AG, POST and the drain all shift by
  that much: about -0.3 to -0.6 µs per shape (bigger relative gain on the narrow shapes, ~2-4%). Judge with
  push_s - rin_end and F_COLLECT end in tl.py.
- Risk: matmul per tile (32x32x32 HiFi4) costs more FPU than the eltwise square. If it exceeds the per-core input
  arrival rate (~100 ns/tile) PRE would lag the read and eat the gain. Expected ~50-70 ns/tile, so it should keep up.
- Accuracy: the diagonal sum is at least as precise as the old reduce (which unpacked fp32 S to tf32). A wrong
  transpose orientation (x^T x) would give column sums -> large PCC failure, a clear signal. Stale cache reads on
  BRISC are avoided with `invalidate_l1_cache()` (fence on BH) before and after the in-place gather.
