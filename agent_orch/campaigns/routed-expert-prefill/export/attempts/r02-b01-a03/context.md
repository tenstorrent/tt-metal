## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules and allowed paths.
- device/kernels/compute/fused_swiglu.cpp: the whole kernel. GU K-loop with reserve-once partials
  and L1_ACC, SiLU copy pass, binary_activation_phase, multiply_phase, down matmul_phase, and the
  chunk loop with the RE-* zones.
- device/unified_routed_expert_ffn_program_factory.cpp:
  - the gu subblock width (largest divisor of pcN <= 8, so 6 on I=2048 and 3 on I=2880/3072)
  - in0_block_w_gu (16, or the largest divisor of K that fits L1: gpt-oss 15, kimi-k3 8)
  - DST_CAPACITY 8 (fp32 dest acc off)
- tt_metal/hw/inc/api/compute/{situ_glu.h, clamped_silu_glu.h, compute_kernel_api.h (silu_tile vs
  silu_tile_pack: identical calculate_silu template), matmul.h (matmul_block_init has no PACK part),
  tile_move_copy.h (copy_init), bcast.h (add_bcast_rows_init), reconfig_data_format.h, pack.h
  (pack_tile<true> index is relative to the pointer moved by cb_push_back)}.
- tt-llk blackhole:
  - llk_math_eltwise_binary_sfpu{,_params}.h: the SFPU uses only ADDR_MOD_7 for these ops, and
    sfpu_start sets the dst addr from the issuing thread's dest_offset_id.
  - cmath_common.h: dest_section_flip stalls CFG on MATH|SFPU1, so a MATH commit waits for the
    pack-thread SFPU to go idle.
  - llk_unpack_AB_matmul.h: one streamed-operand unpack per tile-mult.
- ckernel_sfpu_situ_glu.h, ckernel_sfpu_clamped_silu_glu.h, moe_gpt swiglu_sfpu.h: per-element SFPU
  cost (tanh/exp/recip polynomials). There is no cheap bit-identical speedup, so the lever is overlap.
- moe_gpt/device/kernels/compute.cpp and matmul bmm_fused_activation.hpp (apply_activation_from_pack):
  the established pattern for running the SFPU on the PACK thread.

## Nodes consulted
- All 24 nodes (every reflection; the proposals of the scheduling nodes).
- r02-b01-a02 (parent): the per-expert ACT/GU/DN zone table, the main motivation.
- r02-b04-a02: the same phase split; "activation is the measured, untried lever"; "don't touch DM
  ordering of the down stream".
- r01-b02-a04, r01-b03-a03, r01-b04-a03: subblock geometry and per-call overhead are neutral, so a
  narrower last-block subblock is safe.
- r01-b04-a02: in0_block_w_gu is an accuracy knob. Unchanged here.

## Docs / external references
- None beyond the code above.
