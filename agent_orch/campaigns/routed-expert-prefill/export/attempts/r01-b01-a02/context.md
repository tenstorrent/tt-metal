## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: the protocol, allowed paths, accuracy gate.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: CB sizing and the L1 fit (per_core_M_max vs in0_block_w_gu), sender placement, runtime/CT arg layout, the per-group-size topologies.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp: the runtime group picker (pick_group_rows / expert_cost / GroupScheduler) from r01-b01-a01.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the gate/up loop. The fixed x sender per row reads RM sticks and multicasts bf16, and every receiver pushes cb_x_rm. Also the phase-4 activated multicast (INCL_SRC + write-barrier pattern, reused here).
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp: matmul_phase_fused_gu runs tilize init -> tilize -> uninit -> matmul_block_init -> pack reconfig -> L1_ACC restore on EVERY gate/up K-block on every core, and the down/activation phases.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT / IN1_WRITER_MCAST protocol, and the writer draining cb_out before the next expert's up reads (a possible later target).
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h, llk_math_matmul.h: reuse_a = ct_dim >= rt_dim. Every subblock shape needs >= 1 operand unpack per tile-mult, so taller subblocks do not cut unpack work (contrary to r01-b02-a01's suggestion 3).
- ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp: fast_tilize is used for bf16 in half-sync.
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: LoFi ideal is 16 cycles/tile, and BH bf4 LoFi big matmuls reach ~80% utilisation.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: weight ND-shard layout (1 tile-row x per_core_N shards, round-robin over banks).
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h, tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp: MEM_MAP_END = 40960 + 69 KB kernel config. That gives an L1 CB budget of ~1379 KB, used to check that the fitted (M, w) does not change on any shape.

## Nodes consulted
- r01-b01-a01 (parent, 1.4376): row groups. Its profile log (`reports/r01-b01-a01/.logs/profile_log_device.csv`) gave per-group math-end times. Groups doing identical work finish 5-20% apart at random from run to run (mistral 152/165/198 us min/median/max). The fastest group still runs at ~28-32 cycles per tile-mult against the LoFi 16, so the per-group pipeline is compute-efficiency-bound, not DRAM-bound (mistral ~95 GB/s per group, gpt-oss 147 GB/s total).
- r01-b02-a01 (1.4017): independent row-group variant. Its x DRAM-read rotation gave ~3% on the K=128 shapes. G=1 on kimi-k3 matched R=2, which suggests an R=1 option plus tile-weighted balancing could help kimi-k3 (noted for later).
- r01-b03-a01, r01-b04-a01 (0.94, 0.93): weight read-ahead and read/multicast overlap on the sender RISCs regressed every shape. Both reflections point at the compute's per-K-block fixed cost (tilize init/uninit, matmul re-init, pack reconfigs), which is what this node attacks.

## Docs / external references
- None beyond the in-repo files above.
