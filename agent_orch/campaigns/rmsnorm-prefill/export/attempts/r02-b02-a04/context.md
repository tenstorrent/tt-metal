## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — the job and the allowed_paths glob
  (`dit_fused_distributed_rmsnorm/*`, fnmatch `*` also matches `/`, so the whole op directory is allowed).
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push, the go wait, the 8 x 64 B DRAM stick
  reads after go, and the dual-NoC drain.
- ttnn/.../dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read only, outside allowed_paths) —
  F_COLLECT, F_FABRIC, then a serial `group_go_sem.up()` per worker. It is shared with GroupNorm, so it was copied
  into the op dir as dit_rmsnorm_forwarder.cpp instead of being edited.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — the forwarder kernel creation, worker/forwarder
  virtual coords, forwarder RT arg layout, and override_runtime_arguments (it touches only rt[0], rt[1]).
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — the stats scratch is a caller-owned persistent
  mesh-coherent DRAM buffer, and the out_ready GlobalSemaphore is ping-ponged by the caller. This is deliberate
  cross-call / cross-chip isolation, which is why this node does NOT move the AG landing zone into an L1 CB.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — the post-AG combine (add ring tiles, transpose_dest,
  *1/H, +eps, rsqrt, pack) and the prescale_weight single POST pass.
- tt_metal/hw/inc/api/semaphore.h, api/dataflow/semaphore_dm_impl.h, internal/dataflow/dataflow_api_addrgen.h,
  internal/tt-1xx/blackhole/noc_nonblocking_api.h — Semaphore::set_multicast, and the multicast addr encoding (no
  coordinate mirroring on BH, so the NoC1 start/end swap is the caller's job, as in the matmul mcast factories).
- tt_metal/hw/inc/api/dataflow/endpoints.h — PrecomposedUnicastEndpoint for reads from precomputed NoC addrs.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml, UMD blackhole_coordinate_manager.cpp,
  tt_metal/hw/inc/experimental/drisc_mode.h, tt_metal/llrt/metal_soc_descriptor.cpp — checked whether the drain
  could write to a DRAM bank's other two NoC ports to spread link load. It can't: firmware puts every
  non-preferred DRAM NIU in stream mode (traffic ends in DRISC L1, not GDDR). Dead idea, recorded for others.

## Nodes consulted
- All 22 committed nodes' reflections (r01-*, r02-*).
- r02-b02-a03 (parent) — `tl_out.txt` gave the per-phase timeline. Its reports were re-analysed per core with
  /tmp/r02b02a04/go.py. Findings: go arrives 0.17 µs (slot 0) to 0.52 µs (slot 19) after F_FABRIC end, TRISC end
  tracks go, and the late y=3 row drains last. W_DRAIN start - go = 0.65-0.72 µs (stick read).
- r02-b02-a01 — dual-NoC drain plumbing (DM_DYNAMIC_NOC writer), NoC labels on BH (writer = NoC1), and its
  "y=3 ~0.3 µs later" observation.
- r02-b02-a02 / a03 — the PRE-tail work. It is kept unchanged and is orthogonal to this node.
- r01-b03-a01/a02 — dispatch cost of more kernel groups and RT args. This node adds only ~11 forwarder RT words on
  one core and no new kernel group.

## Docs / external references
- None beyond the in-tree headers above.
