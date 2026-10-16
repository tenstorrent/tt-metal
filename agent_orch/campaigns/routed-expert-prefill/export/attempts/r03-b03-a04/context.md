## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, accuracy gate.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp (best node r03-b02-a03's copy) —
  matmul_phase (down), matmul_phase_fused_gu (gate/up K-loop + pipelined last block), kernel_main inits.
  Every matmul ran rt = 1 (out_subblock_h = 1), ct = subblock width.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp — subblock widths
  (largest divisor of per_core_N <= 8; ragged down tail), bf16 dst (8-tile half), LoFi, per_core_M_max.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h — reuse_a = ct >= rt: in0 held in SrcB,
  in1 streamed into SrcA (one SrcA unpack per mult); reuse_b streams in0 through SrcB with a
  TILE_SIZE_B * KT_DIM step (KT_DIM from init), holds the in1 tile in SrcA.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h — dst layout t + rut*ct for reuse_b,
  CLR_A / CLR_B per reuse mode; MOP depends only on the init's reuse decision.
- tt_metal/hw/inc/api/compute/matmul.h — matmul_block / matmul_block_init / BH dynamic throttle wrapper.
- tt_metal/tt-llk/.../llk_unpack_common.h, llk_unpack_common_api.h — 2-arg reconfig_data_format is
  unconditional and refreshes TILE_SIZE_B, so the reuse_b in0 stride is right for x and activated.
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md — BH LoFi GEMMs reach up to 96% of 16 cycles/tile.
## Nodes consulted
- All 44 nodes' reflections (r01-r03). Key ones:
- r03-b02-a03 — HW counters: 16 FPU + 17.5 SrcA-unpack cycles per tile-mult, serialized; proposed
  streaming x instead of the weights (this node).
- r03-b03-a03 (parent) — accuracy_fail from the SiLU-GLU product; code base reverted to r03-b02-a03.
- r03-b02-a02, r03-b01-a01 — kernel config buffer limit on h7168 shapes; size must not grow.
- r01-b02-a04 — subblock width at rt = 1 doesn't move the cost; factory note: 1x1 down subblocks cost
  +28% op time (per-call overhead), hence reuse_b only at per_core_M >= 3 (>= 6 mults per call).
- r03-b04-a03, r02-b02-a03 — DRAM-bound shapes absorb compute savings; expect gains only on
  compute-bound shapes (mistral, minimax-m3, gpt-oss, dsv3/glm phase A).
## Tools
- /tmp/w_r03b03a04/sz (copy of r03-b02-a03's local TRISC compile harness pointed at this worktree):
  base vs new text bytes, dsv3-like SiLU CTA under the profiler: T0 19464 -> 20708, T1 14632 -> 13724,
  T2 14956 -> 13892 (net -728 B). gpt-oss (SwiGLU-OAI + bias) and TILE-x variants also compile.
