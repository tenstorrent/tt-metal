## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: rules, metric, allowed paths.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: routing (seed 0, uniform top-k),
  the weight ND-shard spec comes from ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID.x (so GRID_X must stay 11).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: routed_expert_weight_memory_config (shard = 1 K-row x
  per_core_N, round-robin over banks). Also passes UNIFIED_ROUTED_EXPERT_CORE_GRID to moe_fused_swiglu, so kCoreGridY
  must not change.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: grid, CB fit, sender placement,
  per-group-size topology args, RT arg layout.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp: kCoreGridX/Y=11x8 (also included by
  moe_fused_swiglu and hybrid types).
- kernels/adaptive_chunk.hpp: chunking, expert_cost, GroupScheduler, pick_group_rows (8/4/2).
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp, ..._writer.cpp, kernels/compute/fused_swiglu.cpp: group
  selection, topology indexing, x owner rotation, down activated multicast, mailbox protocol.
- moe_gpt swiglu_sfpu.h, ckernel_sfpu_situ_glu.h: activation cost (SFPU-heavy on gpt-oss/minimax/kimi-k3).
- reports/r01-b03-a03/.logs/profile_log_device.csv header: 1350 MHz; worker cores at physical y=2..11 on 11 columns,
  i.e. an 11x10 compute grid, of which the op uses 11x8.
- reports/r01-b03-a03/run.log: per-shape counts_min/max.

## Nodes consulted
- r01-b01-a01 / r01-b02-a01: row-group expert parallelism, the base of the current schedule and cost model.
- r01-b01-a02 / r01-b02-a03: row-shared x tilize (current x path); gpt-oss is compute-bound at ~5% from tilize.
- r01-b02-a02: x-sender rotation and 1-row groups hurt; kept groups >= 2 rows.
- r01-b01-a03: pacing failed; group tail imbalance not fixable that way.
- r01-b03-a01/a02, r01-b04-a01/a02/a03, r01-b03-a03: in-pipeline DM timing, ring depth, K-width (accuracy knob),
  boundary feed all neutral/negative -> chose a work-partition lever instead.

## Docs / external references
- /tmp/rep_b03a04/sim.py: Python replica of tiles_of / expert_cost / GroupScheduler over the test's routing, per layout.
