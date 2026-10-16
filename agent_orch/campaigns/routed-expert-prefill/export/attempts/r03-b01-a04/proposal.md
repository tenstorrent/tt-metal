# r03-b01-a04: stream x through SrcB instead of the bf4 weights through SrcA: at per_core_M >= 3 the non-last gate/up K-blocks and the down K-loop call matmul_block with rt = per_core_M, ct = 1 (reuse the weight tile, stream x), bit-identical

## Motivation
- r03-b02-a03 (measurement node, perf counters on MATH) found what sets the ~30-40 cycles per tile-mult on the
  compute-bound shapes (mistral, minimax-m3, gpt-oss, and dsv3/glm phase A):
  - FPU busy is exactly 16.1 cycles/mult (the LoFi ideal).
  - SrcA unpacker (unpacker 0, the streamed bf4 weight tile) is busy 17.5-18 cycles/mult.
  - The two barely overlap: FPU 50-55% + UNP0 55-62% of the window, ≈ 31-33 cycles/mult total. MATH_SRC_DATA_READY
    tracks FPU busy, so MATH is waiting for the next SrcA tile.
  - FW matmul throttle never engaged (0 of ~500k samples).
- Today every matmul here is a 1 x w subblock (`out_subblock_h == 1`): `ct >= rt` -> LLK `reuse_a`. The x tile sits in
  SrcB for w mults and a new bf4 weight tile is unpacked into SrcA for EVERY mult.
- With `rt > ct` the LLK switches to reuse-B: the weight tile is unpacked into SrcA once and x tiles are streamed into
  SrcB (`_llk_unpack_AB_matmul_` `!reuse_a` branch, address step `TILE_SIZE_B * KT_DIM`). That is the mode the BH
  big GEMMs run in (rt 4 x ct 2), and they reach 80-96% of the 16-cycle LoFi ideal, including bf4 x bf4.
- So this is the direct lever on the measured bottleneck, r03-b02-a03's next #1. Nobody has run it.
- On this lineage (r03-b01-a03) the planner is done and the binaries have ~12 KB of kernel-config headroom.

## Mechanism
Compute kernel only (`kernels/compute/fused_swiglu.cpp`), JIT, no host change.
1. `matmul_phase_fused_gu`, non-last K-blocks, when the runtime `per_core_M` (EFF_M) >= 3 and num_blocks > 1:
   - `matmul_block_init(x, gate, 0, ct=1, rt=EFF_M, in0_block_w)` (reuse-B MOP) at phase start and after an owner's
     x tilize; the last K-block re-inits to the 1 x LW config the PACK-thread activation pipeline uses.
   - The real output columns are walked in chunks of W = 8 / EFF_M columns (one dst half). Gate and up each get one
     acquire per chunk. Inside it, column j runs `matmul_block(x, w, k, k * pcN + n, dst = j * EFF_M, 0, ct=1,
     rt=EFF_M, kt)` over the block's K-tiles, then tile (m, n) is packed from dst `j * EFF_M + m` to partials slot
     `m * pcN + n` (the same absolute slot the 1 x w path packs).
2. `matmul_phase` (down), same rule: EFF_M >= 3 -> reuse-B init, the real columns (phantom subblocks and the ragged
   tail respected) in chunks of 8 / EFF_M, real-K bound `k_steps` unchanged.
3. per_core_M 1-2 keep today's code exactly (at rt = 2 one call is only 2 mults and the per-call TRISC overhead may
   eat the gain; it is also mostly the DRAM-bound many-expert shapes).

Numerics: each dst tile still accumulates the block's K-tiles in ascending order from a zeroed dst, with the same
MVMUL replay (only the final CLR_A / CLR_B differs), and is packed with the same L1_ACC sequence into the same slot.
x stays in SrcB and the weight in SrcA. So the result must be bit-identical.

## Why this is not a repeat
- r01-b02-a04 changed the subblock WIDTH at rt = 1 (1x3 -> 1x5 + 1x4): still streaming weights through SrcA, < 1%.
- r03-b04-a03 / r02-b01-a03 hide the activation; r03-b03-a03 changes the SiLU math. None changes which operand is
  streamed through the unpacker.
- Every DM node (rings, read-ahead, VCs, x owner, trids) attacks delivery, which the counters say is not the limiter
  on these shapes.

## Expected effect and risk
- If SrcB streaming overlaps the FPU as in the big GEMMs: the reuse-B K-blocks go from ~31 toward ~18-22
  cycles/mult. Compute-bound shapes: gpt-oss (pcM 4-5, GU 41 cyc/mult) -10..-20%, mistral / minimax-m3 -5..-15%,
  dsv3 / glm a few %. DRAM-bound shapes (kimi-k2.7, dsv4-pro, kimi-k3) ~0 (they have pcM <= 2 or wait on DRAM).
- If SrcB streaming does not overlap either: neutral to slightly worse (more acquires / calls per block). The MATH
  zones RE-GU / RE-DN separate the two phases.
- Risks: wrong dst/slot mapping or a TILE_SIZE_B / KT_DIM GPR mismatch -> PCC change or accuracy_fail (bit-identical
  numerics are the check); per-call TRISC overhead at rt = 3 (3 mults per call).
