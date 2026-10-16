## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the job, rules and allowed paths.
- $DREAM_HOME/routed-expert-prefill/history.md: the index of all 48 nodes.
- Every node's reflection.md (via `git show <tag>:.../reflection.md`), read in full for r01-r03.
- device/kernels/compute/fused_swiglu.cpp:
  - the matmul_phase / matmul_phase_fused_gu K-loops (1 x W subblocks, rt = 1 everywhere);
  - the five matmul_block_init sites, which every unpack-op -> matmul transition passes through;
  - the owner tilize helper (rewrites SCRATCH_SEC0).
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h: the stock MOP. Its per streamed tile body is
  UNPACR, RDCFG, ADDDMAREG(TILE_SIZE_A), STALLWAIT(STALL_CFG, THCON), WRCFG, NOP. There are two halves (config
  contexts 0/1) selected by the TT_MOP zmask. The execute writes the context base addresses per call and runs
  TT_MOP(ct-1).
- llk_math_matmul.h (BH): 16 MVMULs per tile from replay; per tile SETC16 dst offset + MOP. Nothing in it waits
  on unpack config.
- llk_lib/experimental/llk_unpack_AB_custom_mm.h, llk_math_custom_mm.h: the DeepSeek decode matmul. Its SrcA
  stream is UNPACR; CFGSHIFTMASK(Base_address += SCRATCH); NOP per tile. It is limited to tiny in0 tiles, so it
  can't be used directly.
- llk_unpack_tilize.h (BH):
  - It uses the same two-context CFGSHIFTMASK replay with SCRATCH_SEC0.
  - Its init preloads SCRATCH_SEC0 through SETDMAREG + STALLWAIT + WRCFG, the in-order pattern copied here.
  - The owner tilize in this op therefore clobbers SCRATCH_SEC0, so the matmul re-init must rewrite it.
- llk_pack_untilize.h: the CFGSHIFTMASK replacement for "ADDDMAREG + STALLWAIT + WRCFG + NOP".
- llk_unpack_common.h, ckernel_gpr_map.h: TILE_SIZE_A semantics. The units are 16 B, the same as the CB
  fifo_page_size the API passes.
- hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h, hw/inc/api/compute/matmul.h: the operand swap
  (in1 -> SrcA = SEC0) and the tile sizes from fifo_page_size.
- tt-llk docs/performance_counters/performance_counters.md: the counter banks (for reading RE-HW).

## Nodes consulted
- r03-b02-a03 (parent): the RE-HW counters (31 cyc/mult, FPU 16.1, UNP0 busy 17.5 per mult, MATH_SRC_DATA_READY
  tracks FPU). They are the before-numbers for this node, parsed by /tmp/r03b02a03/hw.py.
- r03-b02-a04, r03-b03-a04, r03-b01-a04: operand swap / reuse-B. They halved unpack work and cyc/mult was
  unchanged, so the gap is per streamed tile, not unpack bandwidth. That points at the per-tile MOP body.
- r01-b02-a04: subblock width (per-call overhead) is not it.
- r03-b02-a02: the kernel-config-buffer margin on this lineage. TRISC0 was cut ~4.5 KB there, so +0.7 KB here
  is fine.
- r03-b04-a03 / r03-b04-a04 / r02-b02-a03: which shapes are DRAM-bound (no compute gain expected there).

## Tooling
- /tmp/r04b01/sz/kc.sh: copy of r03-b02-a03's local TRISC compile harness, pointed at this worktree.
  - base/ is the parent kernel, new/ is this node.
  - TRISC0 .text is 18768 -> 19444 B.
  - The profiler build (-DPROFILE_KERNEL=1) compiles on all 3 TRISCs.
