## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — job definition, allowed paths, gate.
- history.md and every node's reflection / score (r01..r03), proposals of r02-b02-a03, r03-b04-a02.
- kernels/compute/fused_swiglu.cpp (parent and r03-b02-a02 versions) — matmul_phase_fused_gu's K-loop, the
  last-block activation walk (enter_act / leave_act / act_tiles / pop_pending / skip_pending), multiply_phase.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop protocol (UP_SPLIT, IN1_WRITER_MCAST,
  row-shared x owner, x look-ahead) and the phase-4 down loop (per-block ack -> read -> mcast chain).
- device/unified_routed_expert_ffn_program_factory.cpp — ring depth (gu_slots / d_slots 2-3), up_mode, semaphores
  (15 in use), CB sizing.
- kernels/weight_runs.hpp — ND-shard coalesced reads.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — weight memory config (1-tile-row ND shards, round-robin
  banks) and LoFi compute config (fp32_dest_acc_en=False, so the dst half is 8 tiles).
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h — dest_section_flip waits on SFPU1 (why one SFPU chunk
  overlaps exactly one MATH acquire).
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h, llk_api/llk_unpack_AB_matmul_api.h — unpack
  address = fifo_rd_ptr - 1 + index * page_size in uint32 (wrapped ring offsets resolve correctly).
- llk_io/llk_io_unpack.h, circular_buffer_init.h — fifo_limit = start + size (16 B units); a pop may not straddle the
  limit (pop the walked blocks one at a time).
- ckernel_sfpu_silu.h / sigmoid / exp / recip (BH) — SiLU cost; r02-b04-a04's op-local SiLU body (not used here).

## Nodes consulted
- r03-b02-a02 (best, 1.5952) — base code for this node; its RE-GU/ACT/DN zones per row (mistral / gpt-oss tails).
- r03-b02-a01, r02-b01-a03, r02-b03-a04 — the one-block PACK pipeline and its measured gains.
- r03-b04-a01 / a02 (parents) — planner trim and x-trid reader; RE-GW block cadence (gate wait vs block period).
- r02-b02-a03, r01-b01-a03, r03-b03-a02 — DM / pacing attempts and why they were neutral or negative.
- r03-b01-a01 / a02 — kernel config buffer limit (70656 B) on h7168 shapes.

## Docs / external references
- Profiles: reports/r03-b02-a02/.logs/profile_log_device.csv (per-row GU/ACT/DN zones), reports/r03-b04-a02 (RE-GW
  per gate block on senders). Scratch analysis was inline python, not kept.
- Local TRISC compile harness /tmp/routed-r03b04a03 (kc.sh from /tmp/r03b02a02/sz, pointed at this worktree; run.sh compiles
  3 TRISCs with hand-built dsv3 / gpt-oss / kimi-k3-like CT args): old vs new object sizes +1.0-1.5 KB, non-fused
  (odd subblock width) and num_blocks 1 / 2 variants compile.
