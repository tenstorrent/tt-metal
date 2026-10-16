## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, metric, allowed paths
- kernels/adaptive_chunk.hpp: planner (layouts, schedule_in_order / schedule_lpt, two-phase make_plan); used to replay each shape's plan (/tmp/r06b01a01/pt.cpp, from /tmp/w_r04b03a03c/sim)
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: gate/up loop, issue_gate_read (gate + up head rows on NoC 0), look-ahead, x owner path
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: UP_SPLIT up read of rows [up_head_rows, w) on NoC 1, writer multicast
- kernels/weight_runs.hpp: shared read helper (also included by hybrid_routed_expert_ffn kernels; new helper is unused there)
- unified_routed_expert_ffn_program_factory.cpp: NoC balance note, up_head_rows = (w+2)/5, d_head_tiles, RE_UP_HEAD_ROWS define for reader and writer
- tests/ttnn/nightly/.../test_routed_expert_prefill.py: routing (token counts per expert per shape, /tmp/r06b01a01/counts.py)
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM endpoints per NoC (noc0 rows 11/2/9/5, noc1 rows 1/10/4/7 on x=0,9)
- tech_reports/Saturating_DRAM_bandwidth: FCFS within a VC, route overlap congestion

## Nodes consulted
- r05-b02-a01 (parent, best 1.6667): the 60/40 NoC split; its profile is the main evidence (per-row first-expert GU gradient)
- r04-b02-a01 (50/50 split): same per-row table, milder gradient -> the split moves group speed by row
- r05-b01-a01: plan-once + tie pass (other candidate; not ported here)
- r05-b03-a01: x(0) arrival gradient by grid row (top rows late), owner spread didn't fix it
- r05-b04-a01: wrap vs no-wrap multicast waits; hang with sender-row moves
- r02-b02-a02: request VCs by (column + group): neutral; rows mixed in every VC
- r03-b03-a02, r02-b04-a02: whole stream on one NoC loses
- all r01-r04 reflections (history.md + git show) for the dead ends (handshakes, look-ahead, ring depth, LLK)

## Analysis scripts (not committed)
- /tmp/r06b01a01/zones.py <profile_log_device.csv> <shape>: per-row RE-GU/ACT/DN start+duration per expert
- /tmp/r06b01a01/percore.py: per-core table of one zone instance
- /tmp/r06b01a01/ends.py: per-row TRISC1 kernel end
- /tmp/r06b01a01/tilt.py, fair.py: link-load model with the profiler's columns (2-7,10,11,13-15)
