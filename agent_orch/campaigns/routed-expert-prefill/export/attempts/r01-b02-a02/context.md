## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, metric, allowed paths.
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp: the parent's `group_rows` / `GroupDealer` schedule.
- .../kernels/dataflow/unified_routed_expert_ffn_reader.cpp: x sender (fixed vs the parent's per-K-block rotation), the in0/in1 handshake order (an x sender that is also the weight sender does x read, then mcast, then gate read, serially), and the FUSE_BIAS per-expert bias read (unconditional in the parent).
- .../kernels/compute/fused_swiglu.cpp: the expert loop and the unconditional bias pop at expert end.
- .../kernels/dataflow/unified_routed_expert_ffn_writer.cpp: up-sender role, group rectangle (no x role, so unchanged).
- .../unified_routed_expert_ffn_program_factory.cpp: L1 fit (per_core_M_max, in0_block_w_gu), host x sender at column (gy+1)%GRID_X, weight sender gx%GRID_Y, and the TT_FATAL that they never collide.
- tests/.../test_routed_expert_prefill.py: shapes, bias only on gpt-oss, ROW_MAJOR x.
- reports/r01-b02-a01 and reports/r01-b01-a01 profile_log_device.csv: per-core kernel end times (my script /tmp/rep02a02/percore.py). gpt-oss: parent groups end at ~425/460 us, b01's at ~385/385 us.
- the parent's run.log: per-shape counts_min/max and weight GB/s.

## Nodes consulted
- r01-b02-a01 (parent): row groups with runtime x-sender rotation, 1-row groups and round-robin deal. Slower than b01 on gpt-oss (+25%) and dsv4-pro (+6%).
- r01-b01-a01: the same idea with the fixed x sender, groups >= 2 rows and a list-scheduling cost model. Best node. The comparison localises the parent's losses.
- r01-b03-a01, r01-b04-a01: read-ahead / pipelining on the sender RISCs regressed. Anything that delays a sender's multicast costs the whole column, which is the same reason a collision is expensive.

## Docs / external references
- none
