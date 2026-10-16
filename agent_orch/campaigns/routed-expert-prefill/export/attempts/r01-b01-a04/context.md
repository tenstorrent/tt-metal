## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: job, rules, allowed paths, gate.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp: whole kernel. Row-shared
  tilize look-ahead (tilize block b+1 before matmul b; block 0 before the loop), activation phases
  (silu+mul, binary SFPU act), down phase with copy pass, UNPACK->MATH/PACK mailbox expert scan.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: expert
  and chunk loops, x RM read lambda, the synchronous up-front read of blocks 0/1 per chunk, gate/up
  sender/receiver protocol, writer-multicast handshake, down phase.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp: up/down
  split reads, IN1_WRITER_MCAST multicast (no guard for 0 receivers), group topology args.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp: group size picker, cost model,
  GroupScheduler (copyable, used for look-ahead).
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: L1 fit
  (per_core_M_max 4, kimi-k3 8 with w=8), subblock choice and its 1x7 vs 1x3 measurement note,
  DOWN_SPLIT "issue-bound" note, group topology generation and the x/weight collision TT_FATAL.
- moe_gpt compute.cpp and swiglu_sfpu.h: the SFPU-on-PACK-thread pattern (considered, not used).
- ckernel_sfpu_silu.h, ckernel_sfpu_situ_glu.h: per-tile SFPU cost estimate of the activations.
- tt-llk blackhole llk_unpack_AB_matmul.h: reuse_a = ct >= rt, so one streamed unpack per tile-mult.
- tests/.../test_routed_expert_prefill.py + model configs: per-shape dims. I replayed the
  routing (seed 0) in /tmp/r01b01a04/counts.py to get every shape's per-expert tile counts, and the
  schedule in /tmp/r01b01a04/sched.py. The static group loads are balanced on all shapes. R=1 for
  kimi-k3 would put 4 experts on some rows and 3 on others (bytes imbalance), so I dropped it.
- reports/r01-b01-a02/.logs/profile_log_device.csv, analysed with /tmp/r01b01a04/rows.py and grp.py:
  per-group TRISC1 end times per call. The slow group is random per call. In calls where two groups
  finish ~15% before the other two, the slow pair does not appear to accelerate afterwards.

## Nodes consulted
- r01-b01-a01/a02 (row groups, row-shared tilize): the base. a02 is the best node, so this node
  reverts a03 to it.
- r01-b01-a03 (pacing, 1.0363): reverted. Its diff shows a per-block non-posted grid multicast plus
  a tight 2-block slack. Combined with the "slow groups don't accelerate" observation, pacing looks
  flawed, not just badly implemented.
- r01-b02-a01..a03: G=1 measured equal to R=2 on kimi-k3, x rotation collisions, "stop porting".
- r01-b03-a01..a03: DM read timing is neutral. b03-a03 names the compute-side boundary critical
  path (down tail, copy pass, x(0) tilize+multicast); this node targets the x(0) part.
- r01-b04-a01..a03: wider K-blocks are an accuracy knob (don't); 3-slot rings neutral.

## Docs / external references
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md (LoFi 16 cycles/tile ideal).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py routed_expert_weight_memory_config
  (ND shard = 1 K-row x per_core_N, ROUND_ROBIN_1D across banks).
