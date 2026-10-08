## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the job, the allowed paths (the op dir glob covers
  subdirs, so a forwarder fork inside `device/kernels/` is allowed).
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: the stick push (write barrier + arrival inc), the go
  wait, the gathered-stick DRAM read at page(d, fwd, round) + slot*128, and the drain. With max_rounds == 1 every
  worker uses packet_buf[0] / page round 0, so splitting slots into waves needs no layout change.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp: the trid-pipelined resident input read (lookahead 4 blocks).
  It is the only input path on the AG resident config, so it is where the wave-B gate and the wave-A signal go.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (DST-accumulated x*x, ones*S^T), x*gamma under the AG,
  the r03-b04-a01 row-0 combine, and the single POST pass. Unchanged by this node.
- dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp: the per-round collect -> fused fabric
  write+inc -> out_ready wait -> go incs. Forked into the op dir as dit_rmsnorm_wave_forwarder.cpp (it is shared with
  GroupNorm and lies outside allowed_paths).
- ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp
  (fused_write_atomic_and_advance_local_read_address_for_fabric_write, perform_payload_send) and
  tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp: the EDM applies the fused inc as
  `noc_semaphore_inc(addr, val)` with a full 32-bit `val`. That makes 16-bit per-wave fields in out_ready safe.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: core allocation (20 workers row-major + 1 forwarder),
  semaphores, reader/writer/forwarder CT and RT args, and the stats page geometry (page = 34 sticks, 1 page per
  device per forwarder).
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py: seq 640, so 20 tile rows and 20
  workers on every shape; num_links 1, so one forwarder.
- tt_metal/hw/inc/api/semaphore.h, dataflow_api.h (noc_semaphore_wait_min): the Semaphore<> API, get_semaphore, and
  invalidate_l1_cache in poll loops.

## Nodes consulted
- All 27 reflections. The key ones:
- r02-b02-a04: the AG section is ~3 µs; the stick read after go is ~0.67 µs; pack's POST is blocked behind the drain.
- r03-b01..b04-a01: the combine gap is now 0.38-0.43 µs. Kernel end moves 1:1 with drain start, so the drain is
  throughput-bound from its first tile. All four r03 branches' "next" lists point at the same remaining combine gap,
  so this node deliberately takes a different direction.
- r01-b02-a02, r01-b03-a02: reads reach ~400 GB/s with 20 or 80 cores, so reads look aggregate-bound. That is the
  premise that 10 cores might also saturate, untested.
- r02-b04-a01, r02-b01-a01, r02-b03-a01: placement and dual-NoC changes. Link sharing between co-located cores matters
  in both dimensions, which is why wave A is the even workers (spread over both rows) and not one row.
- r01-b03-a01: per-call dispatch and cross-chip skew interact with op length. Reading per-chip results needs care.
