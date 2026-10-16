## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, allowed paths, metric.
- $DREAM_HOME/routed-expert-prefill/history.md — whole index (65 nodes, r01-r06).
- unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp — planner, layouts, sender placement
  (`sender_row_in_group`: corner senders since r06-b04-a01). Considered a planner rebalance for
  kimi-k3 and rejected it: 28 experts on 5 groups need 6 on three groups whatever the tie-break,
  and r06-b02's per-row ends show y6-7 (6 experts) in the slow cluster, so flipping ties moves
  little.
- unified_routed_expert_ffn_program_factory.cpp — NoC balance note, d_head_tiles / up_head_rows,
  writer runtime-arg layout (phase-A sems last; remap words appended after them).
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — the two NoC-1 weight read sites (UP_SPLIT
  up rows, DOWN_SPLIT tail tiles), PHASE_A_SEM_RT.
- kernels/weight_runs.hpp — WeightRuns::read; ported r06-b03-a01's DramPortRemap overload.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml — `dram:` port rows and `dram_views`
  worker_endpoint [noc0, noc1] per channel: NoC-0 ports [2,0,0,0,2,2,2,2], NoC-1 port 1 for all.
  So the x = 9 column's NoC-0 tiles are rows 11/3/8/6, not the x = 0 rows the r05/r06 link models used.
- tt_metal/llrt/metal_soc_descriptor.cpp — Metal logical DRAM core (view, i) orders a bank's tiles as
  [NoC-0 endpoint, NoC-1 endpoint, remaining subchannel]: logical (b, 2) is the free tile. Used on
  the host to find each bank's free port without hardcoding it.
- tt_metal/distributed/mesh_device.cpp dram_sender_logical_cores — confirms the ordering
  ("pins the NOC1 endpoint to y == 1", "free non-endpoint subchannel").

## Nodes consulted
- r06-b04-a01 (root, 1.6931) — corner senders, bottom-row down multicast on writer NoC 1.
- r06-b03-a01 (1.5712) — nearest-of-3 NoC-1 DRAM tile remap + ~50/50 split: DN -7..-8%, GU +10-17%;
  its next #1-#3 (down-only, never the NoC-0 tile, one knob) shaped this node. Its diff is ported.
- r05-b02-a01 (1.6667) — 60/40 split; NoC-1 column links were the limit; link model noc4.py.
- r06-b01-a01, r06-b02-a01 — per-row speed gradient is set by others' NoC-0 traffic / first-mover;
  lowering the NoC-0 share is consistent with that.
- r05-b04-a01, r05-b01-a01 — wrap diagnosis; plan-once headroom (not used here).
- r01..r04 summaries from history.md (handshake / read-ahead / operand-reuse directions closed).

## Docs / external references
- /tmp/r05b02a01/noc4.py, /tmp/r06b04/nocw2.py, /tmp/r06b03a01w/m/ep*.py — earlier link models.
- /tmp/r07b03/model.py — this node's model: r06-b04-a01 corner placement + per-position down
  multicast NoC, correct x = 9 NoC-0 endpoints, NoC-1 tile modes def / free / nearest, per-port load.
- tt_metal/hw/firmware/src/tt-1xx/brisc.cc, hw/inc/internal/firmware_common.h,
  internal/dataflow/noc_address_backend_xy.h — the DRAM bank -> NoC xy table lives in each RISC's
  local memory, is loaded once at firmware boot, and TensorAccessor DRAM addresses come from
  `dram_bank_to_noc_xy[noc][bank] << NOC_ADDR_COORD_SHIFT`, so toggling the writer's NoC-1 row
  around a read loop retargets exactly those reads.
- tt_metal/impl/device/firmware/risc_firmware_initializer.cpp — table entries are
  ((y << 6) | x) << reg_offset (0 on BH) from get_preferred_worker_core_for_dram_view(channel(bank)).
- tt_metal/impl/allocator/allocator.cpp — bank_id -> dram channel is identity with one bank per channel.
- reports/r07-b03-a01/run.log (first eval) — "Program size (70784) too large for kernel config
  buffer (70656)" on kimi-k2.7 with the per-run compare remap.
- compute/re_hwprof.hpp — profiler-only MATH counters; made opt-in for kernel-config headroom.
