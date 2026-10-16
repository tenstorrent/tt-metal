## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, metric, gate.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp (full): gate/up phase (1xN
  subblocks, L1_ACC, row-shared tilize), activation phases, down phase with copy-out into the
  2-subblock cb_out.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (full): counts broadcast, group schedule,
  per-block x/weight handshakes, down-phase act multicast. This is where pacing hooks in.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp (full): up/down split reads, IN1 writer
  mcast, cb_out drain order (drain before next chunk's up reads).
- kernels/adaptive_chunk.hpp: group picker, expert_cost, GroupScheduler (reused for per-group totals).
- unified_routed_expert_ffn_program_factory.cpp: CB sizing / L1 fitter (cb_out is 2 subblocks;
  dsv4-pro has ~17 KB of L1 left), semaphore list (13 of 16 used, hence a CB board rather than
  4 new semaphores), runtime-arg layout (counts-broadcast rectangle is on every core).
- tt-llk blackhole llk_unpack_AB_matmul.h / llk_math_matmul.h: reuse_a = ct >= rt, so in1 is
  re-unpacked per row anyway. Taller subblocks only save per-call overhead, and the factory note
  (1x7 -> 1x3 costs 1%) says that overhead is small, so I rejected the subblock idea.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h, semaphore_dm_impl.h: noc_semaphore_set_multicast
  (non-posted 4 B), get_noc_multicast_addr. risc_common.h: get_timestamp_32b.
- tests/.../test_routed_expert_prefill.py: output is TILE bf8, so a down-partials direct drain
  (bf16) would need a format change. That ruled out the "skip copy-out" idea.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: ND-shard weight layout
  (1 K-row x per_core_N shards, round-robin banks).

## Nodes consulted
- r01-b01-a01, r01-b01-a02 (parent): group mechanism; a02's reflection flags group imbalance as the
  largest remaining prize.
- r01-b02-a01/a02: placement collisions, 1-row groups bad.
- r01-b03-a01, r01-b04-a01: read-ahead inside a sender's pipeline loses, so this node avoids it.
- r01-b03-a02: x batching loses.
- r01-b04-a02: in0_block_w_gu is an accuracy knob (bf16 dst), so don't widen it.

## Measurements made for this node (from reports/r01-b01-a02)
- Per-group (2-row band) TRISC1 end times per measured call: random 3-15% slowest/mean
  imbalance, no systematic row position.
- DRAM traffic incl. x and output: dsv3 ~401 GB/s, kimi-k2.7 ~423, dsv4-pro ~407, glm ~398,
  kimi-k3 ~371, mistral ~347, minimax ~368, dsv4-flash ~367, gpt-oss ~178.

## Docs / external references
- tech_reports/GEMM_FLOPS: LoFi ideal 16 cycles/tile.
