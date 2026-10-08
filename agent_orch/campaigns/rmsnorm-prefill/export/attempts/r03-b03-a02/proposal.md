# r03-b03-a02: PRE row stat straight out of the accumulating DST: fp32 transpose_dest + SFPU column-sum (sfpu_reduce<SUM, REDUCE_COL>) puts sum(x^2) per row into row 0 before the single pack, replacing the S pack -> L1 -> unpack -> ones*S^T matmul -> pack round trip

## Motivation
The AG start is gated by the slowest worker's stick push (F_COLLECT end = max W_PUSH end), and W_PUSH waits on the
row stat. At h7168 R_INPUT ends at ~6.5 µs but W_PUSH ends at ~8.15 µs; the PRE tail after the last input tile is
~1 µs and does not scale with width (r01-b04-a04).
- r02-b02-a02 (stat as the diagonal of a DST-accumulated x*x^T, no S round trip) made the stat ready 0.35-0.6 µs
  earlier (W_PUSH start min, tl.py), but its BRISC diagonal gather gave 0.36 µs back.
- r02-b02-a03 (current lineage: ones*S^T matmul on the packed S) made it only 0.0-0.3 µs earlier than reduce+transpose.
  Its reflection: "the ~1 µs PRE tail is dominated by the S pack -> L1 -> unpack handoff (cross-TRISC CB sync +
  fp32 tile pack/unpack + matmul re-init) ... one extra single-tile FPU stage after that handoff costs only ~0.1 µs."
  Its next-step #1 asks for the stat to come straight out of the accumulating DST with no gather.
So the remaining PRE tail is the S round trip through L1, and nobody has removed it without a BRISC gather.

## Mechanism
`dit_rmsnorm_fused_compute.cpp`, resident whole-row packed-AG PRE (`mm_row_stat` case), Blackhole only:
1. Keep the DST-accumulated S = sum_k x_k∘x_k (ELWMUL dest-MAC into DST 0, fp32).
2. Inside the SAME acquire, before commit:
   - `transpose_dest<fp32>(0)`: S -> S^T in place (the same LLK the POST combine already uses).
   - `sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0)`: column sums into row 0. Column c of
     S^T sums to sum_j S[c][j] = sum(x^2) of token c, so row 0 is exactly the stick layout (face 0 row 0 = tokens
     0-15, face 1 row 0 = tokens 16-31) the writer sends as two 64 B face-rows.
3. Pack DST 0 once to `stats_transposed_local_cb` and push. No pre_intermediate_cb round trip, no matmul init,
   no second unpack/math/pack stage. The unpack thread is free right after the last x*x unpack and can start the
   x*gamma unpacks earlier.
Non-BH builds keep the matmul path. Nothing else changes (writer, reader, factory, POST).

Precision: the row sum is now an fp32 SFPU add tree over the fp32 S, instead of S truncated to tf32 in SrcA for
the matmul. It should be at least as accurate.

## Why this is not a repeat
- r02-b02-a02: also took the stat straight from DST, but left it on the diagonal and gathered it on BRISC (+0.36 µs).
  Here the row-0 layout is produced in DST by the FPU transpose and an SFPU reduce, so the writer is unchanged.
- r02-b02-a03 / root: same S accumulation, but S goes through L1 and a matmul. This removes exactly that handoff,
  which r02-b02-a03 identified as the PRE tail's cost.
- The round-3 siblings (r03-b01/b02/b04) and the parent all work on the post-AG combine. This attempt is on the pre-AG
  critical path (stick push -> AG start), so it is orthogonal and stacks with them.

## Expected effect and risk
- The stat should be ready ~0.3-0.5 µs earlier (W_PUSH start min). transpose_dest<fp32> plus a 4-iteration SFPU
  column reduce should take ~0.15-0.25 µs, against the ~0.6-0.8 µs pack/unpack/matmul/pack chain. F_COLLECT end and the
  AG end should move by ~0.2-0.4 µs on every shape, and everything after them moves with them. Expected about -0.2 to
  -0.4 µs per shape (+1.5-3%).
- Risks:
  - The wrong reduce orientation would give a garbage stat, which would show up as a PCC collapse.
  - The SFPU reduce leaves replay slots [0,9) recorded. Every later SFPU/FPU op re-inits before use, so this should
    not matter.
  - transpose_dest flips the zero-flag/implied-format state, which is the same state the POST combine already
    restores from.
  - If the slowest worker's W_PUSH is gated by its input read rather than PRE, the gain shrinks. tl.py's
    R_INPUT end vs W_PUSH end will show which.
