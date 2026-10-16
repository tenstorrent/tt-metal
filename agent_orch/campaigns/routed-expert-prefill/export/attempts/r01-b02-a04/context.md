## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: the job, rules, allowed paths, accuracy gate.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: shapes, routing, dtypes (bf4 weights, LoFi, RM bf16 x).
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: subblock rule (largest divisor <= 8 -> 1x3 for per_core_N 9), ragged down support, cb_out sizing, ring checks, L1 fitter, valid-subblock runtime args.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp: fused gate/up phase (1 x w subblocks, one matmul_block call per k), down matmul_phase (already ragged-aware), activation phases, SiLU copy + multiply_phase.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp: row-group rule (rows = 2 for most shapes, 4 for gpt-oss), the GroupDealer, CB ring alignment rules.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (first ~600 lines): DRAM read NoC/VC usage (noc_read(0), default request VC), bias reads, x owner reads.
- unified_routed_expert_ffn/device/kernels/weight_runs.hpp: ND-shard coalesced weight reads.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h: per call, per reused-dim step t: wait_for_next_context, address config, semaphore post, STALLWAIT, srcB unpack, MOP of ct srcA unpacks (each with a CFG STALLWAIT), semget, context switch. MOP run count is passed per call; only reuse_a = ct >= rt is baked at init.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h: the math replay depends only on reuse_a; ct/rt are runtime per call.
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_situ_glu.h, ckernel_sfpu_softcap.h: SiTU-GLU cost (two tanh polys + exp + recip per element); activation is a serial per-expert phase (considered and rejected as a lever: fixed numerics).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: ND-shard weight layout (1 tile-row shards, round-robin over banks).
- tech_reports/Saturating_DRAM_bandwidth, tech_reports/GEMM_FLOPS: LoFi ideal 16 cycles/tile; VC/FCFS fairness notes (considered: per-group read-request VCs as an alternative idea).

## Analysis done (scripts in /tmp/r01b02a04)
- counts.py: per-shape per-expert tile counts for seed 0 (dsv3 [6,5,6,5,5,5,5,6], kimi-k3 28 x 3-4, dsv4-pro 12 x 3, gpt-oss [8,9,9,8], ...). M-row padding is already optimal at tile granularity on every shape.
- tail.py / rows.py / grp.py over reports/r01-b02-a03/.logs/profile_log_device.csv: mean vs max TRISC1 end per shape (tail 2-10%), per-group finish spread per call (mistral groups 147-197 us in one call; the slowest group changes call to call; cores inside a group finish within a few us).
- Per-core cycles per tile-mult (mean TRISC1 end x 1.35 GHz / per-core tile-mults): kimi-k2.7 36 (1x6), dsv4-pro 41.5 (1x3), kimi-k3 45 (1x3, w=8), mm3 30.5 (1x3, M3), dsv3 29 (1x6, M3), gpt-oss ~40 (1x3 both phases). kimi-k2.7, dsv4-pro and kimi-k3 have the same DRAM bytes per tile-mult (~140 B).
- L1 fitter replica (fit.py, adapted from r01-b04-a03's): the new down rule only changes gpt-oss (3x3 -> 5+4, cb_out 6 -> 18 tiles); its (M, w) = (4, 15) is unchanged with >200 KB spare.

## Nodes consulted
- all 12 nodes' proposal/reflection (r01-b01-a01..a03, r01-b02-a01..a03, r01-b03-a01..a03, r01-b04-a01..a03).
- r01-b04-a02: in0_block_w_gu is an accuracy knob, so don't touch it (this node doesn't).
- r01-b03-a03 / r01-b04-a03: suggested ragged 1x5+1x4 for gpt-oss, and showed that taller subblocks don't cut unpacks.
- r01-b01-a03: pacing for the group tail failed on overhead; this node takes the compute-rate route to the tail instead.
