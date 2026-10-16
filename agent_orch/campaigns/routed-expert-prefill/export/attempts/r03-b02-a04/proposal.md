# r03-b02-a04: column-streaming matmul groups: at per_core_M >= 3 the non-last gate/up K-blocks and the down K-loop hold each bf4 weight tile in SrcA and stream the per_core_M x tiles through SrcB (ct < rt LLK mode) instead of re-unpacking a weight tile per tile-mult

## Motivation
- The parent (r03-b02-a03) measured the per-tile-mult cost with Tensix perf counters on MATH.
  - On the compute-bound shapes a mult costs ~30-33 cycles: mistral, minimax-m3, dsv3/glm phase A, and gpt-oss DN (gpt-oss GU ~41).
  - FPU busy is 16.1 cycles/mult on every shape, the LoFi ideal.
  - The SrcA unpacker is busy 17.5-18 cycles/mult. FPU busy and unpacker-0 busy add to ~105-115% of the window, so they barely overlap.
  - FW matmul throttling is ruled out: the throttle word was never set.
- Every matmul in the op runs 1 x W subblocks (`ct >= rt`, the LLK's `reuse_a` mode):
  - one in0 tile (x / activated, bf8) is held in SrcB;
  - the bf4 weight tile is streamed through SrcA once per tile-mult (`llk_unpack_AB_matmul.h`: the MOP runs ct SrcA unpacks per SrcB unpack).
- With `ct < rt` the LLK switches mode: SrcA (the weight tile) is unpacked once per column, and the rt in0 tiles of that column are streamed through SrcB, stepping by TILE_SIZE_B * KT_DIM.
  - Big BH GEMMs run this way (default subblock (4,2)).
  - The bf4 LoFi entries in tech_reports/GEMM_FLOPS reach ~80% of the 16-cycle ideal (~20 cycles/tile all-in), against ~31 here.
- Earlier subblock attempts (r01-b02-a04's 1x5+1x4; the "taller subblocks" notes in r01-b03-a03 / r01-b04-a03) all kept `ct >= rt`. So the weight stayed the streamed operand.
  - None of them changed which operand is streamed. That is the lever the parent's counters point at (parent's next #1).

## Mechanism
`kernels/compute/fused_swiglu.cpp` (kernel-only, no host change):
- **New out-of-line helpers:**
  - `mm_init`: matmul_block_init, plus pinning TILE_SIZE_B to the in0 tile size for CS shapes.
  - `mm_kloop`: the K-loop of matmul_block calls.
  - `mm_group`: one dst acquire: K-loop, then pack the rows x ct tiles to absolute partials slots `m * per_core_N + n`.
  - `mm_block`: all groups of one K-block, for one or two in1 operands.
- **Column streaming (CS) groups:** used when the runtime per_core_M >= 3 (`kCsMinRows`) and per_core_M <= one dst half (8).
  - A group is all per_core_M rows x cs_ct columns.
  - cs_ct is the widest ct with ct < rows and rows * ct <= 8: 2 at pcM 3-4, 1 at pcM 5-8.
  - Applied to the non-last gate/up K-blocks (gate then up per column group, same x rows) and to every down K-block.
  - At per_core_M 1-2 the old 1 x W walk runs. It now goes through the same helpers, with the same calls in the same order and the same slots.
- **The last gate/up K-block is unchanged.** It is still walked in 1 x LW subblocks, because the PACK-thread activation pipeline is built on that walk. It gets the 1 x LW init back after CS blocks.
  - After an owner tilize (which restores the 1 x W init), the CS init is re-issued lazily.
- **Bit-identical numerics.** Each output tile accumulates the same K-tiles in the same order through the same MVMUL replay; only which src register is held changes. L1_ACC across K-blocks is untouched, and so are the formats, fidelity and approximation modes.
- **Kernel-config-buffer budget.** The h7168 programs sit at the 70656 B limit (r03-b01-a01 and r03-b02-a02 overflowed), and an inlined matmul_block costs ~1 KB on MATH.
  - So every matmul_block / matmul_block_init in the kernel now goes through these single out-of-line copies: the last-block `mm_last` and the phase inits too.
  - The parent's profiler-only `re_hwprof.hpp` is trimmed: the INSTRN_THREAD bank (stalls; the SrcA/SrcB-valid waits read 0) and the throttle samples (always 0) are gone. The FPU and unpacker counters are kept, so the record layout is unchanged.
  - Local compile, profiler on, sum of the 3 TRISC .o sizes vs the parent: SiLU -0.5 KB, SwiGLU-OAI -1.0 KB, SiTU-GLU -1.0 KB.
  - UNPACK grows +1.5 KB, because runtime ct/rt keeps both LLK modes. MATH -0.9..-1.4 KB, PACK -1.1 KB.

## Why this is not a repeat
- r01-b02-a04 (1x3 -> 1x5+1x4) and the "taller subblocks" discussions kept `ct >= rt`, so the bf4 weight was still unpacked per mult. Here the weight is unpacked once per per_core_M mults, and x/activated tiles are streamed instead.
- No node has run any matmul in the `ct < rt` mode.
- r03-b02-a03 is the measurement that motivates this. Its own next #1 proposes exactly this change. I add the down phase in the same mechanism, and a pcM >= 3 gate because of per-call overhead.

## Expected effect and risk
- **Shapes with per_core_M >= 3** (mistral, minimax-m3, gpt-oss, dsv3/glm phase A; kimi-k3 only if 1-row layouts give pcM 3-4):
  - If SrcB streaming overlaps the FPU the way big GEMMs suggest, GU non-last blocks and DN fall from ~30 toward ~18-22 cycles/mult.
  - That is 20-35% off GU+DN on compute-bound cores: mistral / mm3 / gpt-oss -10..-25%, dsv3/glm a few % (DRAM-bound in phase B).
  - DRAM-bound shapes at pcM 1-2 (kimi-k2.7, dsv4-pro, dsv4-flash): unchanged, apart from slightly more RISC work per matmul call through the runtime-parameter helpers.
- **Risks:**
  - The SrcB stream may not overlap either: neutral result, and the trimmed counters (UNP1 busy, FPU util) will say why.
  - Per-call UNPACK bookkeeping at 3 rows per context round may cap the gain.
  - A wrong mode switch (stale CS MOP in the last block, or a missing re-init after tilize) would show as wrong PCC or a hang.
  - The TILE_SIZE_B GPR is pinned explicitly to avoid a stale SrcB stride.
  - Kernel config buffer overflow should not happen (net smaller than the parent).
- **How to read the eval:**
  - PCC / mean_abs must be bit-identical to the parent.
  - RE-GU / RE-DN zones and RE-HW cyc/mult on mistral, mm3 and gpt-oss.
  - SrcA writes per mult should fall by ~per_core_M x, and SrcB writes per mult should rise.
