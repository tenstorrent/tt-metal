## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: job, gate, allowed paths.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp: kCoreGridX/Y = 11/8, a
  fixed rectangle; the export UNIFIED_ROUTED_EXPERT_CORE_GRID feeds moe_fused_swiglu's grid in
  models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py (kept unchanged, see proposal).
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: GRID_Y is used for
  chunk_M_tiles = per_core_M * GRID_Y, the core loop, whole-column / per-group topologies
  ({GRID_Y, GRID_Y/2, GRID_Y/4}), the counts multicast rectangle and its num_dests. The L1 fit
  doesn't depend on GRID_Y.
- device/kernels/adaptive_chunk.hpp: kGridY drives pick_group_rows (rows /= 2 from kGridY),
  group_rows_index, kMaxGroups and GroupScheduler. 10 -> 5 -> 2 falls out of the same loop.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp, _writer.cpp, compute/fused_swiglu.cpp:
  every row/group quantity derives from kGridY / group_rows / my_gy (grep for my_gy, kGridY,
  group_rows). No other hard-coded 8 in kernel logic. The static_asserts chunk_M_max == per_core_M_max *
  kGridY need the define in every kernel.
- hybrid_routed_expert_ffn: has its own adaptive_chunk.hpp and kernels with a UNIFIED_RE_GRID_Y define,
  so it is unaffected by the unified op's header change.
- reports/r01-b04-a03/.logs/profile_log_device.csv: physical rows y=10,11 have only 704 zone rows
  (other ops) against 13574 for each of y=2..9. The device worker grid is 11x10, so 22 cores sit idle.
- reports/r01-b04-a03/run.log: per-shape counts_min/max, DRAM GB/s.
- tests/.../test_routed_expert_prefill.py: _route_tokens (seed 0 uniform top-k). I replicated it in
  /tmp/r01b04a04/sim.py with the kernels' cost model to predict group picks at 8 vs 10 rows.

## Nodes consulted
- r01-b01-a01 / r01-b02-a01 / r01-b02-a02: row-group schedule, cost model, sender placement rules.
- r01-b01-a02 (best, 1.4716) and r01-b04-a03 (parent, same code + 3-slot rings): base code.
- r01-b03-a03, r01-b04-a03 reflections: gpt-oss is the compute-bound outlier, 9-tile experts on
  4-row groups waste one row in four. DM timing, ring depth and K-width are ruled out.
- r01-b01-a03: cross-group pacing failed (don't touch group coupling).

## Docs / external references
- none beyond the code.
