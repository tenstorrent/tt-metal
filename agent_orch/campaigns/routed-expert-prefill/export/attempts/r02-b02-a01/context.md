## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — job, rules, leaderboard (best r01-b04-a04 1.5011, this node's base).
- kernels/adaptive_chunk.hpp — the whole runtime schedule: chunking, expert_cost (units + chunks),
  GroupScheduler (least-loaded, expert order), pick_group_rows (uniform 10/5/2). Rewritten here.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — where the group size, my_group/my_mt,
  topology slot and the per-expert ownership test are derived; the per-expert loop; every use of
  is_in1_sender / in1_num_receivers is inside the expert loop, so an idle row never touches its set.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — same derivation, NoC-1 up/down feeds, output
  row mapping row0 = chunk*chunk_M + my_mt*per_core_M.
- kernels/compute/fused_swiglu.cpp — UNPACK picks the schedule and mailboxes the group height and the
  per-expert tile counts to MATH/PACK; whole compute pipeline read to understand the per-expert
  fixed cost (activation phase, copy passes, L1_ACC drains).
- device/unified_routed_expert_ffn_program_factory.cpp — L1 fit (per_core_M_max 4, in0_block_w_gu),
  sender placement (weight sender row base + gx % rows, x sender col (gy+1)%11), per-group-size
  topology sets (replaced by per-layout sets).
- tests/.../test_routed_expert_prefill.py — routing (seed 0, uniform top-k) to replay counts.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h — TRISC local memory 4 KB (the plan is
  4 B per expert in static storage).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — weight ND-shard layout (1 tile-row x
  per_core_N shards, round-robin over banks).

## Nodes consulted
- r01-b04-a04 (base) — 10-row grid; its reflection measured per-row end times: gpt-oss rows balanced but
  -20% M-units bought only -8% time => large per-expert fixed cost; dsv4-pro heavy groups 3 experts.
- r01-b03-a04 — same 10-row idea via layouts; per-expert fixed cost on gpt-oss; "add layouts after
  simulating the cost model on the test routing".
- r01-b01-a01/a02/a03/a04, r01-b02-a01..a04 — row groups, row-shared tilize, pacing (failed),
  x look-ahead (neutral), subblock shapes (neutral). Showed DRAM-bound big shapes (~410-434 GB/s).
- r01-b03-a01/a02, r01-b04-a01/a02/a03 — read-ahead / batching / wider K / deeper rings all
  neutral or negative; in0_block_w_gu is an accuracy knob. => avoid DM timing tweaks.

## Analysis scripts (not committed)
- /tmp/r02b02/counts.py — replays the test's seed-0 routing: per-shape tile counts
  (dsv3/glm [6,5,6,5,5,5,5,6], kimi-k2.7 3-5 x12, kimi-k3 3-4 x28, mistral/mm3 [5,6,6,5],
  dsv4-flash 4-5 x8, dsv4-pro 3 x12, gpt-oss [8,9,9,8]).
- /tmp/r02b02/sim.py, picker.py — layout/cost models; only dsv4-pro and gpt-oss change picks.
- /tmp/r02b02/cc/t.cpp, fuzz.cpp — host unit test of the new header: 200k random count vectors on
  8- and 10-row grids, uniform-layout picks and owners identical to the old scheduler; mixed
  layouts only chosen when the modelled makespan is strictly smaller.

## Docs / external references
- none beyond the code.
