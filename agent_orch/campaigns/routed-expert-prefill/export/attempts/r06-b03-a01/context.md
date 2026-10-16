## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, accuracy gate.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml — each DRAM bank has 3 NoC tiles (rows {0,1,11},{2,10,3},{9,4,8},{5,7,6} on x=0 and x=9); worker_endpoint NoC1 = port 1 for every bank; the SYS-1419 note pins only NoC-0 endpoints.
- tt_metal/hw/inc/experimental/drisc_mode.h — NIUs come up in NOC2AXI (any tile forwards DRAM-range reads to GDDR); only DRISC firmware puts non-endpoint NIUs in stream mode.
- tt_metal/api/tt-metalium/experimental/tensor_prefetcher.hpp, tt_metal/distributed/mesh_device.cpp (pick_unused_dram_logical_core / dram_sender_logical_cores), tt_metal/impl/buffers/tensor_prefetcher_manager.cpp (IsTensorPrefetcherSupported = DRISC available && no streaming profiler) — the 3 tiles are 3 MPFE ports of one GDDR controller; the public guard for "DRISC may own the other NIUs".
- tt_metal/impl/device/firmware/risc_firmware_initializer.cpp (generate_device_bank_to_noc_tables) — dram_bank_to_noc_xy[noc][bank] = translated preferred endpoint, packed (y<<6)|x.
- umd blackhole_coordinate_manager.cpp (eval checkout) — translated DRAM coords (17 + bank/4, 12 + 3*(bank%4) + port), UMD port order = yaml order; Tensix translation keeps y.
- tt_metal/hw/inc/api/tensor/tensor_accessor.h, internal/dataflow/noc_address_backend_xy.h, blackhole noc_nonblocking_api.h — NoC address = packed_xy << 36 | local; NOC_ADDR_COORD_SHIFT = 36.
- tt_metal/hw/inc/api/dataflow/endpoints.h, noc.h — PrecomposedUnicastEndpoint for raw 64-bit NoC addresses.
- kernels/weight_runs.hpp, kernels/dataflow/unified_routed_expert_ffn_writer.cpp — the two NoC-1 weight read sites (UP_SPLIT up rows, DOWN_SPLIT tail tiles).
- device/unified_routed_expert_ffn_program_factory.cpp — NoC balance note, up_head_rows / d_head_tiles, writer runtime args layout.
- reports/r05-b02-a01/run.log — FW bundle 19.7.0 on this machine (< 19.12, so no DRISC).

## Nodes consulted
- All 61 nodes' reflections (history.md + /tmp/r06b03a01w/refl.md); proposals of round 5 in full.
- r05-b02-a01 — the weight stream is NoC-link-bound; NoC-1 column links hot; its link model (/tmp/r05b02a01/noc*.py) is the base of mine.
- r05-b04-a01 — multicast wrap finding (NoC-1 gate/up mcast, NoC-0 down mcast) — kept out of this node.
- r05-b03-a01 — x-owner column spread was null; row gradient of first-block reads.
- r04-b03-a01 / r04-b04-a02 / r04-b03-a02 — block periods set by the weight stream's share; handshake work is done.
- r03-b03-a02, r02-b04-a02 — whole stream on one NoC loses; keep both NoCs.

## Docs / external references
- tech_reports/Saturating_DRAM_bandwidth (cited by r05-b02-a01): route overlap of DRAM responses congests the NoC.
- Link model: /tmp/r06b03a01w/m/ep.py, ep2.py (with weight multicasts), ep3.py (2- vs 3-tile choice).
