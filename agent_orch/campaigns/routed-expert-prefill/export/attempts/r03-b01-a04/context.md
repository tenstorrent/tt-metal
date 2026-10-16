## Files read
- `agent_orch/WORKER.md`, `agent_orch/campaigns/routed-expert-prefill/campaign.yaml`: rules, gate, allowed paths.
- `kernels/compute/fused_swiglu.cpp`: `matmul_phase` (down K-loop, ragged/phantom subblocks, real-K bound,
  partials slot = m * N_PER_ROW + n), `matmul_phase_fused_gu` (non-last K-blocks with 1 x out_subblock_w
  gate and up acquires; owner tilize re-init; last-block PACK-thread activation walk at 1 x LW), `kernel_main`
  (per-chunk reconfig + matmul_block_init, RE-GU / RE-ACT / RE-DN MATH zones).
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`: reuse_a = ct >= rt. reuse_a keeps in0 (x) in
  SrcB and unpacks one in1 (weight) tile into SrcA per MOP iteration; !reuse_a unpacks the in1 tile once per t
  and streams in0 tiles into SrcB, stepping the SrcB address by TILE_SIZE_B * KT_DIM.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h`: the LoFi MOP depends only on reuse_a (last MVMUL
  clears A or B); dst write address for !reuse_a is dst_index + t + rut * ct. Init re-establishes the zero flag.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h`, `tt_metal/hw/inc/api/compute/matmul.h`:
  matmul_block_init / matmul_block / dynamic throttle wrapper (reads the FW word, re-inits only on change).
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h`: TILE_SIZE_A/B GPRs are set by hw_configure and by
  every srca/srcb data-format reconfig, so TILE_SIZE_B follows the SrcB operand's format (x and activated: bf8).
- `device/unified_routed_expert_ffn_program_factory.cpp`: bf16 dst (DST_CAPACITY 8 per half), gu_out_subblock_w =
  largest divisor of per_core_N_gu <= 8 (6 for pcN 6, 3 for pcN 9), down ragged rule, per_core_M_max 4 (8 when
  widened).
- `tests/.../test_routed_expert_prefill.py`: routing -> ~5-8 tiles per expert, so per_core_M ~2-5.
- `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`: bf4 x bf4 LoFi GEMMs reach ~80% (WH table) / up to 96% on BH, i.e. the
  unpack of the streamed operand overlaps the FPU in the big-GEMM subblock configs (rt > ct streams SrcB).

## Nodes consulted
- All round-3 nodes in full (proposal + reflection): r03-b01-a01..a03 (this lineage: plan-once, planner done),
  r03-b02-a01..a03, r03-b03-a01..a03, r03-b04-a01..a03; history.md index for rounds 1-2.
- r03-b02-a03: the hardware-counter measurement this node acts on (16 FPU + 17.5 SrcA-unpack cycles per mult,
  serialized; throttle never engaged). Its next #1 is this mechanism.
- r01-b02-a04: subblock width change at rt = 1 (< 1%), i.e. width alone does not change the streamed operand.
- r03-b04-a03 / r03-b03-a03: DRAM-bound shapes absorb compute savings (GU+ACT+DN conserved); compute-bound shapes
  are mistral, minimax-m3, gpt-oss (and dsv3/glm phase A).
- r01-b04-a02: precedent that changing K-block geometry (not done here) can fail accuracy.

## Docs / external references
- None beyond the in-tree LLK sources and tech report above.
