## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, metric.
- device/kernels/adaptive_chunk.hpp (parent) — two-phase Plan/PhaseDealer/make_plan, model constants.
- r02-b04-a01's adaptive_chunk.hpp, reader/writer/compute/factory diffs — layout table, plan_groups, per-layout topology sets.
- device/kernels/dataflow/unified_routed_expert_ffn_{reader,writer}.cpp, compute/fused_swiglu.cpp — phase loop, topology arg offsets (GROUP_TOPO_RT, PHASE_A_SEM_RT), owner checks, compute mailbox protocol.
- device/unified_routed_expert_ffn_program_factory.cpp — topology emission and phase-A semaphores (3-way merged cleanly).
## Nodes consulted
- All 20 nodes' score.json + reflection.md (history.md index).
- r02-b03-a01 (parent) — two-phase wins on kimi-k2.7/dsv4-pro, losses on dsv3/glm from two-round phase B; proposed fix #1.
- r02-b04-a01 — best node; mixed (3,3,2,2) for gpt-oss/dsv4-flash; dsv4-pro neutral under mixed.
- r02-b02-a01 — similar mixed-height idea, confirms the gpt-oss gain.
- r02-b01-a01 — fewer, bigger groups lose DRAM concurrency (consistent with the single-round restriction).
## Tools
- /tmp/r02b03a02/mt — 3-way merge (base r01-b04-a04, ours parent, theirs r02-b04-a01).
- /tmp/r02b03a02/pt/pt.cpp — host replay of plan_launch on the test's seed-0 tile counts (from /tmp/r02b03a01/plan_test.cpp).
