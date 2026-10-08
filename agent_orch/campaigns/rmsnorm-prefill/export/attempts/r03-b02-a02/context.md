## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — job, metric, allowed paths
- all 27 nodes' reflection.md (dumped via git show over every tag) — what each mechanism did
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push, go wait, 8 x 64 B gathered-stick
  reads through the `stats_dram` TensorAccessor, drain
- ttnn/.../dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only, outside allowed paths) —
  fused fabric write+atomic to `get_noc_address(stats_dram, page)` on every chip incl. local, then
  out_ready wait + local write barrier, then go incs
- tt_metal/fabric/hw/inc/linear/addrgen_api.h — get_noc_address is accessor-generic (DRAM or L1)
- dit_fused_distributed_rmsnorm.cpp — create_stats_buffer uses make_stats_tensor_spec
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — validate compares buffer_type to the spec
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — make_stats_tensor_spec, compute_sizing (page =
  34 sticks x 128 B = 4352 B, 4 pages), TensorAccessorArgs(stats_dram_buffer) for writer + forwarder
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — combine + POST (unchanged)
- tests/ttnn/nightly/.../test_fused_rms_norm_prefill.py — seq 640 -> 20 tile rows, 1 row per worker; buffers
  created via create_stats_buffer, ping-ponged per call
## Nodes consulted
- r02-b02-a04 — measured go -> stick-in-L1 0.67 µs; pre-staging addresses saved nothing; L1 landing suggested
- r03-b01/b02/b03/b04-a01 — combine gap now 0.38 µs; siblings attack that, so this node goes elsewhere
- r01-b03-a01, r02-b02-a01 — cross-call coupling: shorter op tightens next-call start skew
## Docs / external references
- none
- /tmp/r03b02a02/ag.py (copied to this dir as ag.py) — per-shape AG/stick-read timeline from the profiler CSV
