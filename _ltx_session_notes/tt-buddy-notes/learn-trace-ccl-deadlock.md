# learn-trace-ccl-deadlock

## end_trace_capture hang on 4x8 BH Galaxy with fabric CCL: topology/fabric mismatch is prime suspect
**2026-06-10 04:19** · `tt-metal@9de36f1912f`

**Core insight:** `end_trace_capture` finalize is mostly host-side trace assembly, but it ends by *blocking-writing* the assembled trace into a DRAM mesh buffer (`populate_mesh_buffer` → `enqueue_write_shard_to_sub_grid(..., blocking=true)` → `finish_nolock`), which waits on the completion queue (the `system_memory_manager.cpp:757` timeout). On a Galaxy mesh the captured fabric/EDM CCL programs are configured `Topology.Linear` while the documented physical Galaxy config is Ring/`FABRIC_1D_RING` — that topology↔fabric_config mismatch is the most likely deadlock trigger, exposed under trace because trace requires a fully quiesced, cross-mesh-consistent device state at finalize that eager per-op dispatch never demands.

**Root-cause hypotheses (ranked):**
1. **Topology/fabric_config mismatch (most likely).** CCL ops are hard-coded `topology=ttnn.Topology.Linear` (`models/tt_dit/parallel/config.py:149,186,207`; `audio_ops.py` via `ccl_manager.topology`), but `models/tt_dit/models/LTX2.md:86` documents 32-chip Galaxy as Ring / `FABRIC_1D_RING`. Deepwiki confirms Linear needs `FABRIC_1D`, Ring needs `FABRIC_1D_RING`, and a topology/fabric/physical-mesh mismatch hangs *specifically under trace*. dataset_eval conftest sets `FABRIC_1D` — verify the audio run's actual `fabric_config`; if the mesh is ring-wired or set to `FABRIC_1D_RING` while ops say Linear (or vice versa), fabric routing wedges and the finalize completion-write never drains.
2. **Dummy-GO equalization across ETH cores fails the identical-descriptor invariant.** `record_end` partitions the mesh into device-ranges with identical program sets, then prepends dummy GO signals so every device ends with equal `expected_num_workers_completed` / launch-msg wptr — including unicast (ETH) signals (`num_virtual_eth_cores`). It `TT_FATAL`s if ranges don't produce identical `TraceWorkerDescriptors`. On an asymmetric 4x8 CCL graph (edge chips run different EDM programs than interior), the equalization can mis-count ETH go-signals so devices wait forever on a GO that never arrives.
3. **Fabric/Dispatch ordering not captured in MeshTrace.** Per `tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md`: MeshTrace stores no TT-Fabric broadcast / Dispatch_d command-ordering info; traces are device-local, host loops across devices on replay. If `all_gather_async`/`mesh_partition` rely on implicit fabric ordering, that dependency is lost — but note the reported hang is at *capture finalize*, not replay, so this ranks below #1/#2.

**How it works (finalize path):**
- `ttnn.end_trace_capture` → `MeshDeviceImpl::end_mesh_trace` (`mesh_device.cpp:1292`) → `FDMeshCommandQueue::record_end` (`fd_mesh_command_queue.cpp:1148`) then `MeshTrace::populate_mesh_buffer` (`mesh_trace.cpp:49`).
- `record_end` is pure host assembly: compute identical-program device-ranges (restricted to local mesh partition, `:1159`), pack each program's kernel config via `SimpleTraceAllocator` into per-core-type ringbuffers, prepend dummy GO signals for unused nodes (`:1286`), re-patch dispatch commands (`update_traced_program_dispatch_commands`, `:1388`), append to `ordered_trace_data`. No device finish here.
- `populate_mesh_buffer` does the only device touch: blocking shard writes of trace bytes to the DRAM `MeshBuffer` (`:149`) → `finish_nolock` → completion-queue wait → timeout at `system_memory_manager.cpp:757`. A wedged fabric (hyp.1) or mismatched GO counters (hyp.2) means this drain never completes → "device unrecoverable" timeout.
- Triage clean (no stuck waypoint/NoC/in-flight op) fits a host/dispatcher-side wait on a completion that the fabric state will never produce.

**ACTIVE_ETH config-buffer limit (Q3):** `max_size = get_ringbuffer_size(device, ACTIVE_ETH)` (`program.cpp:2453`); the limit (~25600) is the HAL `KERNEL_CONFIG` dev-size for ACTIVE_ETH. EDM/CCL eth programs are structurally large and already near it; watcher instrumentation inflates kernel size and pushes `state.offset` (27616) over, tripping `program.cpp:2455` at program *finalize* (during capture, not record_end). This is a separate marginal-resource symptom, not the deadlock cause, but signals the ETH path is at the edge.

**Trace requirements (Q4):** `trace_region_size` must be non-zero (or dynamic mode); persistent CCL output buffers + global semaphores must be allocated BEFORE `begin_trace_capture` (no alloc/dealloc during trace — `tech_reports/.../llms.md:1404`). The model pre-warms (`vocoder_ltx.py:383-388`: synchronize → warmup `_forward_device` → begin/capture/end), so buffers are pre-cached via `CCLManager.get_ag_ping_pong_buffer` (`manager.py:152`) — that part looks correct. Confirm `num_links` (1-2) is legal for the actual fabric: on a ring-wired Galaxy, `num_links` and topology must match the physical links.

**Known limitation / escalate?** Deepwiki (tenstorrent/tt-metal) states CCL fabric ops with `Topology.Linear`/`num_links` 1-2 on Blackhole Galaxy hanging at `end_trace_capture` (while fine eagerly) is a *known* class of issue tied to topology/fabric_config/physical-mesh mismatch — but it surfaced no specific GitHub issue number in-repo. Treat as a known-pattern config bug first (fix #1), escalate only if topology/fabric_config are confirmed consistent and it still hangs.

**Recommended next checks:**
- Print the run's actual `device_params.fabric_config` and the physical Galaxy wiring; align CCL `topology` + `fabric_config` (Ring→`FABRIC_1D_RING`, or force Linear+`FABRIC_1D` end-to-end). This is the highest-value experiment.
- Try `num_links=1`, single CCL op in isolation under trace to bisect which collective wedges.
- If alignment doesn't fix it, dump per-device `TraceWorkerDescriptor` counts to test hyp.2.

**Key files:**
- `tt_metal/distributed/fd_mesh_command_queue.cpp` — `record_end` (:1148) host trace assembly; dummy-GO ETH equalization; `finish_nolock` (:554) completion wait.
- `tt_metal/distributed/mesh_trace.cpp` — `populate_mesh_buffer` (:49), the blocking device write at finalize.
- `tt_metal/distributed/mesh_device.cpp` — `end_mesh_trace` (:1292) orchestration.
- `tt_metal/impl/dispatch/system_memory_manager.cpp:757` — completion-queue wait timeout (the observed throw site).
- `tt_metal/impl/program/program.cpp:2453-2460` — ACTIVE_ETH kernel-config ringbuffer limit.
- `models/tt_dit/parallel/config.py:149` — CCL ops hard-coded `Topology.Linear`.
- `models/tt_dit/models/LTX2.md:86` — documents Galaxy as Ring / `FABRIC_1D_RING` (the mismatch).
- `models/tt_dit/parallel/manager.py:152,415` — persistent ping-pong buffers + `all_gather_async` wiring.
- `models/tt_dit/models/audio_vae/vocoder_ltx.py:383` — `forward_traced` warmup/begin/end pattern.
