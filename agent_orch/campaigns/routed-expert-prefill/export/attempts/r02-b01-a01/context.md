## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, eval, allowed paths.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp — group-size picker (pick_group_rows), expert_cost, GroupScheduler; the only file changed.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp (expert loop ~1070-1130), dataflow reader (~377-480) and writer (~249-281) — confirm all three kernels call pick_group_rows with the same tiles_of and use expert_cost for the deal; nothing else depends on the model.
- unified_routed_expert_ffn_program_factory.cpp (per_core_N_gu/d, in0_block_w_d = per_core_N_gu, L1 fit) — per-core tile-mults per tile-row = 2*K_gu*pcN_gu + (pcN_gu*GRID_X)*pcN_d, and weight tiles per expert ~= GRID_X x that.
## Nodes consulted
- All 16 round-1 nodes (proposal + reflection). Key: r01-b04-a04 (parent; per-row end times show the 2-group tails on kimi-k2.7/dsv4-pro), r01-b03-a04 (10-row layouts, cost-model replica), r01-b01-a04 (seed-0 tile counts per expert, /tmp/r01b01a04/counts.py), r01-b03-a03 (DRAM-bound big shapes at 410-434 GB/s), r01-b04-a02 (do not widen K-blocks), r01-b01-a03 (pacing failed).
- reports/r02_root_1 profile (baseline 8-row, per_core_M 1 single group: per-expert solo times used for calibration).
## Docs / external references
- /tmp/r02b01a01/sim.py, fit3.py, intmodel.py: fluid DRAM-sharing simulator fitted to measured times; t.cpp: host compile of the exact header to list the picks.
