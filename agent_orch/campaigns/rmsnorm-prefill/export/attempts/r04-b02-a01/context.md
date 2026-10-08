## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, metric.
- $DREAM_HOME/rmsnorm-prefill/history.md — index of all 35 nodes.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push layout (slot*128, two 64 B face-rows),
  go wait, 8 x 64 B gathered-stick reads from the stats scratch after go, drain.
- dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp — collect -> fused fabric write+inc to page
  (my_device) on every chip -> out_ready wait -> 20 serial go incs. Shared with GroupNorm, so it is forked.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — packed-AG combine: wait ring_size gathered tiles, ELWADD
  tiles 0..3, add_rsqrt on row 0, transpose_dest, pack. Tile indices are plain CB tile indices.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — compute_sizing / make_stats_tensor_spec (single source of
  the scratch spec), worker/forwarder placement (row-major prefix, forwarder = core num_workers), CB + kernel args,
  per-forwarder RT args, override_runtime_arguments (rt[0..1] of the forwarder only).
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — persistent buffer validation (layout + memory config).
- dit_fused_distributed_rmsnorm.cpp — create_stats_buffer allocates make_stats_tensor_spec(compute_sizing(...)).
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_api.h — unpack address = fifo_rd_ptr + fifo_page_size *
  tile_index, so a 64 B-page CB lets compute address a tile at any 64 B offset.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h — the tile-size GPR set from fifo_page_size is only
  consumed by the matmul unpack, not by eltwise binary.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h — noc_async_write_multicast_loopback_src /
  noc_semaphore_set_multicast_loopback_src both on NOC_MULTICAST_WRITE_VC (data and flag ordered).
- ttnn/.../matmul/.../reader_bmm_tile_layout_in0_sender_padding.cpp — precedent: data mcast then flag mcast, no barrier.
- tt_metal/fabric/fabric_edm_packet_header.hpp, hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp — fabric has no
  usable NoC multicast at the destination (NOC_MULTICAST_* "has bug", ASSERT), scatter is max 4 chunks: so the fabric
  can't land in 20 workers directly; landing in the forwarder and re-multicasting on-chip is the way.
- ttnn/.../ccl/common/kernels/minimal_ccl_common.hpp — fused_write_atomic... also does the local NoC write of my page.
- tt_metal/fabric/hw/inc/linear/addrgen_api.h — fabric dest address = accessor.get_noc_addr (works for sharded).
- reports/r03-b02-a02/.../ops_perf_results*.csv — host vs device timing (cross-chip launch stagger on h3584/h6144).

## Nodes consulted
- All 35 reflections (r01..r03). Most relevant:
- r03-b02-a02 (parent/best) — go -> stick-in-CB 0.60 us, only 0.09 us of it memory latency; reflection #1 = this idea.
- r02-b02-a04 — go multicast (flag only) and stick-address pre-staging: no gain; serial unicast go spreads 0.17..0.52 us;
  its forked forwarder + rectangle RT args are the plumbing pattern reused here (rows of logical x -> virtual rect).
- r03-b01-a02 — L1 scratch: the fabric landing commit isn't the cost, the worker read handshake is.
- r03-b02-a01 / r03-b03-a01 / r03-b04-a01 — combine gap now 0.38 us (left unchanged here).
- r03-b04-a03 — gamma streaming (likely ported by a sibling; orthogonal to this change).
- r02-b02-a01 / r03-b02-a03 / r03-b03-a03 / r03-b04-a02 — drain mechanisms tried (dual NoC, cmd bufs, VCs, waves).

## Docs / external references
- none beyond the source above.
