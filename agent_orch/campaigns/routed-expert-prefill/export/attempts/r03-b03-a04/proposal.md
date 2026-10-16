# r03-b03-a04: reuse-B matmul at per_core_M >= 3 — hold the bf4 weight tile in SrcA and stream x / activated rows through SrcB (rt = per_core_M, ct = 2) in the non-last gate/up K-blocks and the down K-blocks

## Motivation
- r03-b02-a03 (best node, 1.6103) measured the compute bottleneck with Tensix perf counters: on the
  compute-bound shapes a tile-mult costs ~31-33 cycles = FPU 16.1 (LoFi ideal) + SrcA unpack 17.5,
  serialized (FPU busy ~50%, unpacker-0 busy ~57%, they add to ~105-115%). gpt-oss GU is 41.5 cyc/mult.
- Cause: every matmul in the op runs out_subblock_h = 1 (rt = 1, ct = 3..7), i.e. LLK `reuse_a`
  (ct >= rt): the x tile is held in SrcB and EVERY bf4 weight tile is unpacked into SrcA once per
  tile-mult. That is the operand that does not overlap.
- No node has changed which operand is streamed. Subblock width (r01-b02-a04), ring depth, read-ahead,
  x placement, VCs etc. never moved cyc/mult, consistent with this.
- BH large GEMMs (tech_reports/GEMM_FLOPS: up to 96% of the 16 cycles/tile LoFi ideal, bf4 included)
  run 4x2-type subblocks, i.e. LLK `reuse_b` (ct < rt): the in1 tile stays in SrcA for rt mults and in0
  tiles stream through SrcB. So streaming SrcB can overlap the FPU, and SrcA unpacks drop to 1/rt per mult.

## Mechanism
Kernel-only change, in `unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp` (on the
best node r03-b02-a03's code; this reverts the parent r03-b03-a03's SiLU-GLU accuracy failure):
- Per chunk, pick the matmul block shape from the runtime per_core_M (EFF_M):
  - EFF_M >= 3 (and EFF_M <= 8 = dst half): rt = EFF_M, ct = min(EFF_M - 1, 8 / EFF_M) (3 -> 3x2,
    4 -> 4x2). ct < rt, so the LLK takes reuse_b: weight tile held in SrcA, x (or activated) streamed.
  - EFF_M <= 2: today's rt = 1, ct = out_subblock_w (unchanged path, bit-identical).
- Gate/up K-blocks 0..N-2: one dst acquire per (all rows) x ct-column group, K-loop of
  matmul_block(rt, ct), pack rt*ct tiles to the same absolute partials slots (m*N + n). Gate and up
  keep separate acquires as today. The last K-block (rt = 1 LW walk pipelined with the PACK-thread
  activation) is untouched; it re-inits to rt = 1 first.
- Down K-blocks: same, keeping the real-K bound (k_steps), phantom-column bound and the ragged tail.
- matmul_block_init is issued with the chosen (ct, rt) per chunk, after each owner tilize, and before
  the down phase. One loop serves both shapes (rt = 1 path is the same loop with ct = subblock width),
  and the K-loop is a shared helper, so code growth stays small (h7168 programs are near the kernel
  config buffer limit).
- Numerics: each output tile still accumulates the same K tiles in the same order in dst, then the
  same L1_ACC packs. The mult operands are identical (in0 -> SrcB, in1 -> SrcA). Expect bit-identical
  PCC / mean_abs.

## Why this is not a repeat
- r01-b02-a04 (ragged 1x5+1x4 subblocks) changed the WIDTH of rt = 1 subblocks: still one SrcA weight
  unpack per mult. This node changes the reused operand.
- r01-b01-a04 / r01-b04-a03 / ring and DM nodes changed delivery, not the per-mult unpack.
- r03-b02-a03 proposed exactly this as its next #1; nobody has implemented it.

## Expected effect and risk
- Shapes / phases at per_core_M >= 3: mistral, minimax-m3, gpt-oss (all groups), dsv3 / glm phase A.
  If SrcB streaming overlaps the FPU: GU and DN cyc/mult ~31-41 -> ~20-24 in those windows. mistral /
  mm3 / gpt-oss -5..-15% op time where compute-bound; dsv3 / glm smaller (phase B is pcM 2, DRAM-bound).
  DRAM-bound pcM-1/2 shapes (kimi-k2.7, dsv4-pro, kimi-k3, dsv4-flash) are unchanged by construction.
- Risks: (a) SrcB (bf8) streaming may serialize just like SrcA -> neutral; the RE-HW counters
  (cyc/mult, UNP1 busy) will show it. (b) mistral's GU may already sit at its DRAM share
  (r03-b04-a03), so it could absorb the saving. (c) kernel config buffer overflow on h7168: checked
  locally with the TRISC size harness before the eval. (d) wrong dst/pack indexing -> accuracy_fail on
  the pcM>=3 shapes only.
