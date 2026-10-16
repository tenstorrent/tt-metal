# r01-b02-a04: balanced ragged matmul subblocks: per_core_N 9 runs as 1x5 + 1x4 instead of 3 x 1x3 (gate/up on hidden-3072/2880 shapes, down on gpt-oss)

## Motivation
Every non-gpt-oss shape now sits at 350-430 GB/s of total DRAM traffic. Per row group, a 2-row group's
compute rate is only slightly above its DRAM share (dsv3: a lone group at ~28 cycles per tile-mult
consumes ~100 GB/s of weights; 4 groups ~400 GB/s). So compute and DRAM bind together, and a group that
loses some DRAM share can barely catch up. That is the 3-10% group tail every node since r01-b01-a02 has
measured. In this regime, cutting compute cost per tile-mult raises both the steady-state rate and the
catch-up headroom.

Per-core cycles per tile-mult, from the parent's profile (reports/r01-b02-a03, mean TRISC1 end, per-core
tile-mults counted from the factory's tiling), for shapes with the SAME DRAM bytes per tile-mult
(~140 B; all M=2 per expert):
- kimi-k2.7 (gate/up subblock 1x6, down 1x7): 36 cycles, 434 GB/s
- dsv4-pro (gate/up 1x3, down 1x7): 41.5 cycles, 389 GB/s
- kimi-k3 (gate/up 1x3, down 8+3, w=8): 45 cycles, 351 GB/s
gpt-oss (gate/up 1x3, down 1x3, bias) is at ~40 cycles and 186 GB/s, the most compute-bound shape.
mm3 (1x3 gate/up, M=3) is at 30.5 vs dsv3's 29 (1x6), so the penalty shows mostly at M=2.

The shapes with per_core_N_gu = 9 (hidden 3072 -> 96 tiles, and gpt-oss 90 tiles, over 11 columns)
get gate/up subblocks of 1x3, because the factory takes the largest DIVISOR of per_core_N that is <= 8.
Each matmul_block call is one unpack "t" iteration on BH (llk_unpack_AB_matmul.h: per row it waits for a
free unpack context, programs the addresses, posts the semaphore, does a STALLWAIT, unpacks srcB once, then
runs a MOP of ct srcA unpacks, then does semget + a context switch). So that fixed cost is paid per 3
tile-mults instead of per 6-7. Gate/up is 2/3 of the tile-mults on these shapes.

## Mechanism
Pick the subblock grid with the FEWEST N-subblocks that fit DST (8 tiles), balanced, and allow a ragged
last subblock (the down path already supports this): w = ceil(N / ceil(N/8)), tail = N - (nsb-1)*w.
- per_core_N 9 -> 5 + 4 (2 calls per row per k instead of 3; srcB unpacks 2 instead of 3).
- Use it only when it strictly reduces the subblock count against the existing divisor rule, so every
  other shape keeps its exact grid (6, 7x3, 6x3, 6x2, and kimi-k3's down 8+3 are unchanged).
- Gate/up (factory + compute): add the ragged tail to `matmul_phase_fused_gu`, with
  in1_num_subblocks = ceil, the tail width derived in-kernel from per_core_N_gu, and EFF_OUT =
  M * per_core_N_gu. The non-fused SiLU copy loop and `multiply_phase` now walk EFF_OUT in chunks of
  <= out_subblock_w (flat elementwise ops, so any chunking is valid; the rings stay one block, so pushes
  still end exactly on fifo_limit). The ring checks use the row granule when ragged. The valid-subblock
  runtime arg (ceil(valid/w)) is already correct for a w...w,tail layout.
- Down: the existing ragged support now also applies when it cuts the count (gpt-oss 3x3 -> 5+4). The
  cb_out ring becomes row-sized there (18 tiles), and by the L1 fitter replica gpt-oss still has >200 KB
  spare, so (M, w) do not move.
- The LLK keeps reuse_a = ct >= rt for both widths (rt = 1), so one matmul_block_init (at ct = w) is valid
  for the narrower tail call. The down phase relies on exactly this today.

Files: unified_routed_expert_ffn_program_factory.cpp, kernels/compute/fused_swiglu.cpp.

## Why this is not a repeat
- r01-b02-a01/b04-a02 suggested taller subblocks (h > 1). b03-a03/b04-a03 showed these never reduce
  srcA unpacks, and with reuse_a the LLK still pays the per-row context cost, so h > 1 saves nothing at
  the LLK level. This node instead widens ct, which cuts the per-row LLK cost and the srcB unpacks.
- b03-a03 suggested "ragged 1x5+1x4 gate/up and down subblocks" for gpt-oss but nobody implemented it.
  This applies it to all four per_core_N = 9 shapes, including the two longest (kimi-k3, dsv4-pro).
- Not a DM/read-timing change (b03-a01, b04-a01, b03-a03 are all neutral or negative there) and not a K-block
  width change (the accuracy knob, b04-a02).

## Expected effect and risk
- Numerics: each output tile's K accumulation order is unchanged (k order within the block, L1_ACC
  across blocks), so it should be bit-identical to the parent on all shapes.
- Speed: kimi-k3 / dsv4-pro +3-8%, gpt-oss +5-10% (both gate/up and down), mm3 +0-3%, the other five
  shapes unchanged (identical code path apart from loop bookkeeping). Geomean ~+1.5-2.5%.
- Risks: CB ring misalignment on the ragged path (would show as a hang or garbage output on exactly
  the four shapes), or an LLK assumption about the ct of a call versus the init (would show as wrong
  output). The SiLU / non-fused path with per_core_N 9 is not benchmarked; it is kept correct by
  construction (flat chunking), but is untested here.
