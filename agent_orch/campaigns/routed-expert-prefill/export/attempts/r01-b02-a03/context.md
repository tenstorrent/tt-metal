## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, gate, allowed paths.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp: matmul_phase / matmul_phase_fused_gu
  structure (1 x out_subblock_w subblocks, L1_ACC partials, per-block tilize on the RM path), kernel_main
  GroupDealer mailbox flow.
- .../kernels/dataflow/unified_routed_expert_ffn_reader.cpp: group plan, in1 sender rows, fixed in0
  sender, M-row NoC table (M_ROW_NOC_RT_OFFSET, indexed by grid column), effective_chunks gating, x_blk.
- .../unified_routed_expert_ffn_program_factory.cpp: CB footprint/fitter, CB ids, reader CT args 34-36,
  compute RT args (my_nt_gu == gx).
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h: unpack count for rt x ct matmul is
  rt*ct + min(rt,ct) either way, so taller subblocks don't cut in1 unpacks. Rejected that idea.
## Nodes consulted
- r01-b01-a02: the source mechanism (diff ported with git apply -3; conflicts only in compute RT-arg
  block / group setup and the factory compute args).
- r01-b02-a02 (parent), r01-b02-a01: fixed x sender vs rotation; bias skip for other groups.
- r01-b01-a01: shared schedule baseline for b01-a02's gains.
- r01-b03-a01/a02, r01-b04-a01/a02: per-block DM/handshake reorderings lose; widening w is an accuracy
  knob (forbidden).
## Docs / external references
- none
