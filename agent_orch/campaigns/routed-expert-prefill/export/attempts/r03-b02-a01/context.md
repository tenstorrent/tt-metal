## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, allowed paths.
- device/kernels/adaptive_chunk.hpp: the full planner (plan_launch -> make_plan -> pick_group_rows / model_phase / model_expert; plan_groups -> schedule_in_order / schedule_lpt). Counted where tiles_of and the divides are hit.
- kernels/dataflow/unified_routed_expert_ffn_{reader,writer}.cpp and compute/fused_swiglu.cpp: how each kernel builds tiles_of (L1 counts + idx pages) and calls plan_launch before its first DRAM read.
- kernels/weight_runs.hpp and models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py (routed_expert_weight_memory_config): weight reads are already one NoC transaction per per_core_N shard row. The DRAM-bound shapes sit at ~410-455 GB/s, so delivery efficiency is not the lever.
- tests/.../test_routed_expert_prefill.py: bf4 ND-sharded weights, ROW_MAJOR dispatch buffer.
## Nodes consulted
- Every r02 reflection, plus the r01 index and next-steps in history.md.
- r02-b02-a04 (best, planner-only) and r02-b01-a04 (PACK pipeline + short bodies, compute-only): git diff --stat against their common ancestor r02-b03-a03 shows disjoint files. b01-a04's reader/writer delta is r02-b01-a02's neutral `start` fold and is not taken.
- r02-b01-a04 / r02-b03-a02 / r02-b03-a04: planner prologue measurements (first-GU start 29 µs on kimi-k3) and suggested trims.
- r02-b02-a03 / r02-b04-a02 / r02-b02-a02: DM / stream-timing tweaks that only move bandwidth between groups. Avoided.
## Docs / external references
- /tmp/routed_r02b02a04/fuzz.cpp (r02-b02-a04's planner fuzz) as a template; my old-vs-new equivalence fuzz is in /tmp/r03b02a01/eq.cpp.
