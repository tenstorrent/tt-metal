# metal Runtime (runtime) — Function / Subfunction Decomposition

Functional decomposition with ownership, mapped to the functions in
`runtime-item-definition.md` §7–8 and to requirements in
`runtime-safety-properties.md` / `runtime-functional-capabilities.md`.

Ownership legend:
- **runtime** — owned by metal Runtime.
- **UMD** — owned by the User-Mode Driver (downward).
- **HAL** — owned by the hardware abstraction layer.
- **SFPI** — owned by the external RISC-V JIT toolchain.
- **?** — ownership unconfirmed → recorded in `runtime-baseline-decisions.md`.

Safety-relevant subfunctions marked **[S]**.

---

## F1 — Device & context lifecycle management  (item fn §7 bullet 1)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F1.1 Runtime option / env parsing | runtime | `CODE:tt_metal/llrt/rtoptions.cpp` | FR-01, HLR-06c |
| F1.2 Cluster / topology discovery **[S]** | runtime→UMD | `CODE:tt_metal/llrt/tt_cluster.cpp` | FR-01, HLR-06b |
| F1.3 HAL construction (arch mem map) **[S]** | HAL | `CODE:tt_metal/llrt/hal.cpp`, `CODE:tt_metal/llrt/hal.hpp` | FR-01, ENV-04/05 |
| F1.4 Device open / activation **[S]** | runtime→UMD | `CODE:tt_metal/impl/device/device_manager.cpp` | FR-01, HLR-01a |
| F1.5 Firmware / RISC init on cores **[S]** | runtime | `CODE:tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` | HLR-05b |
| F1.6 Dispatch-topology / CQ-kernel init **[S]** | runtime | `CODE:tt_metal/impl/dispatch/topology.hpp` | FR-08, HLR-05b |
| F1.7 Ready-state gating **[S]** | runtime | `CODE:tt_metal/impl/device/device.cpp` | HLR-05b |
| F1.8 Ordered teardown / cleanup **[S]** | runtime | `CODE:tt_metal/impl/device/device_manager.cpp` (`close_devices`) | HLR-05c |
| F1.9 Context singleton lifecycle **[S]** | runtime | `CODE:tt_metal/impl/context/metal_context.cpp` | HLR-05c |

## F2 — Memory allocation  (item fn §7 bullet 2)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F2.1 Allocator / bank management **[S]** | runtime | `CODE:tt_metal/impl/allocator/allocator.hpp`, `CODE:tt_metal/impl/allocator/bank_manager.hpp` | FR-02, HLR-01d |
| F2.2 Free-list allocation algorithm **[S]** | runtime | `CODE:tt_metal/impl/allocator/algorithms/free_list_opt.hpp` | HLR-01d |
| F2.3 Buffer creation & page mapping **[S]** | runtime | `CODE:tt_metal/impl/buffers/buffer.cpp` | FR-02, HLR-02a |
| F2.4 Reserved-region enforcement **[S]** | runtime | `CODE:tt_metal/impl/host_api/tt_metal.cpp` | HLR-01c |
| F2.5 Overlap validation **[S]** | runtime | `CODE:tt_metal/common/core_coord.cpp` (`validate_no_overlap`) | HLR-01d |
| F2.6 Mesh buffer placement **[S]** | runtime | `CODE:tt_metal/distributed/mesh_buffer.cpp` | FR-02 |
| F2.7 Circular buffer / semaphore alloc | runtime | `CODE:tt_metal/impl/buffers/circular_buffer.cpp`, `CODE:tt_metal/impl/buffers/semaphore.cpp` | FR-05 |
| F2.8 Allocator stats / reporting | runtime | `CODE:tt_metal/impl/memory_tracking/` | FR-16 |

## F3 — Program & kernel build  (item fn §7 bullet 3)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F3.1 Program / kernel object creation | runtime | `CODE:tt_metal/impl/program/program.cpp`, `CODE:tt_metal/impl/kernels/kernel.cpp` | FR-03 |
| F3.2 Kernel compile-hash computation **[S]** | runtime | `CODE:tt_metal/impl/program/kernel_compile_utils.hpp` | FR-04, HLR-02d |
| F3.3 JIT build (toolchain invocation) **[S]** | runtime→SFPI | `CODE:tt_metal/jit_build/build.cpp` | FR-04 |
| F3.4 Build cache / dedup **[S]** | runtime | `CODE:tt_metal/jit_build/jit_build_cache.hpp` | FR-04, HLR-07b |
| F3.5 Generated-source emission | runtime | `CODE:tt_metal/jit_build/genfiles.cpp` | FR-04 |
| F3.6 Precompiled-binary lookup | runtime | `CODE:tt_metal/jit_build/precompiled.hpp` | FR-04 |
| F3.7 Program cache (compiled programs) **[S]** | runtime | `CODE:tt_metal/api/tt-metalium/program_cache.hpp` | FR-04, HLR-07b |
| F3.8 Program-binary status tracking **[S]** | runtime | `CODE:tt_metal/impl/program/program_impl.hpp` (`ProgramBinaryStatus`) | HLR-02d |
| F3.9 Remote JIT compile (optional) | runtime | `CODE:tt_metal/impl/jit_server/` | FR-04 |
| F3.10 Runtime-arg configuration **[S]** | runtime | `CODE:tt_metal/impl/host_api/tt_metal.cpp` (`SetRuntimeArgs`) | FR-06 |

## F4 — Command-sequence construction & dispatch  (item fn §7 bullet 4)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F4.1 Request admission / validation **[S]** | runtime | `CODE:tt_metal/impl/host_api/tt_metal.cpp`, `CODE:tt_stl/tt_stl/assert.hpp` | HLR-01a/01b/02a/06a |
| F4.2 Target / coordinate interpretation **[S]** | runtime | `CODE:tt_metal/llrt/tt_cluster.cpp` (virtual↔umd coord map) | HLR-01b |
| F4.3 Command sizing **[S]** | runtime | `CODE:tt_metal/impl/dispatch/device_command_calculator.hpp` | HLR-03c |
| F4.4 Command sequence construction **[S]** | runtime | `CODE:tt_metal/impl/dispatch/device_command.hpp` | HLR-03c |
| F4.5 Program dispatch assembly **[S]** | runtime | `CODE:tt_metal/impl/program/dispatch.cpp` | FR-08 |
| F4.6 Issue-queue reserve & write **[S]** | runtime | `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` | FR-08, HLR-03c |
| F4.7 Worker-config / launch-msg ring mgmt **[S]** | runtime | `CODE:tt_metal/impl/dispatch/worker_config_buffer.hpp`, `CODE:tt_metal/impl/dispatch/launch_message_ring_buffer_state.hpp` | HLR-03a |
| F4.8 Dispatch-core placement **[S]** | runtime | `CODE:tt_metal/impl/dispatch/dispatch_core_manager.hpp` | FR-08 |
| F4.9 Device-side prefetch/dispatch kernels **[S]** | runtime | `CODE:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`, `CODE:tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` | FR-08 |
| F4.10 Buffer read/write dispatch **[S]** | runtime | `CODE:tt_metal/impl/buffers/dispatch.hpp` | FR-07, HLR-02a/02b |
| F4.11 Mesh workload assembly **[S]** | runtime | `CODE:tt_metal/distributed/mesh_workload.cpp` | FR-08, FR-13 |

## F5 — Completion, event & synchronization  (item fn §7 bullet 5)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F5.1 Completion-queue read / confirm **[S]** | runtime | `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` | HLR-02b/03a |
| F5.2 Event record (barrier + prefetch stall) **[S]** | runtime | `CODE:tt_metal/impl/event/dispatch.cpp` | HLR-03a |
| F5.3 Event id monotonicity **[S]** | runtime | `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` | HLR-03b |
| F5.4 Wait-for-event / Finish / Synchronize **[S]** | runtime | `CODE:tt_metal/api/tt-metalium/distributed.hpp` | FR-09, FR-10 |
| F5.5 Completion-reader thread (mesh) **[S]** | runtime | `CODE:tt_metal/distributed/mesh_command_queue_base.hpp` | FR-09 |

## F6 — Fault detection & status reporting  (item fn §7 bullets 6–7)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F6.1 Precondition / boundary checks (TT_FATAL) **[S]** | runtime | `CODE:tt_stl/tt_stl/assert.hpp` | HLR-01*/02a/06a |
| F6.2 Error propagation to caller **[S]** | runtime | `CODE:tt_metal/impl/debug/watcher_device_reader.cpp` | HLR-04a |
| F6.3 Dispatch operation timeout / hang detect **[S]** | runtime | `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` (opt-in) | HLR-04b, PERF-03 |
| F6.4 Timeout reaction / inspector snapshot **[S]** | runtime | `CODE:tt_metal/impl/context/metal_context.cpp` (`on_dispatch_timeout_detected`) | HLR-04b, HLR-05a |
| F6.5 Watcher device fault sanitization **[S]** | runtime | `CODE:tt_metal/impl/debug/watcher_device_reader.cpp` (opt-in) | HLR-04c, FR-15 |
| F6.6 Host NOC address sanitize **[S]** | runtime | `CODE:tt_metal/llrt/sanitize_noc_host.hpp` (opt-in) | HLR-01b |
| F6.7 Launch-message validation **[S]** | runtime | `CODE:tt_metal/impl/debug/watcher_device_reader.cpp` | HLR-03a |
| F6.8 DPrint / Inspector observability | runtime | `CODE:tt_metal/impl/debug/dprint_server.cpp`, `CODE:tt_metal/impl/debug/inspector/` | FR-16 |
| F6.9 Dispatch telemetry | runtime | `CODE:tt_metal/impl/dispatch/` (telemetry) | FR-16 |

## F7 — Trace capture & replay  (item fn §7 bullet: trace)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F7.1 Trace capture (record nodes) **[S]** | runtime | `CODE:tt_metal/impl/trace/trace_node.hpp`, `CODE:tt_metal/distributed/mesh_trace.cpp` | FR-11 |
| F7.2 Trace assembly / commit to DRAM **[S]** | runtime | `CODE:tt_metal/impl/trace/trace_buffer.hpp` | FR-11 |
| F7.3 Trace replay dispatch **[S]** | runtime | `CODE:tt_metal/impl/trace/dispatch.hpp` | FR-11 |
| F7.4 Trace-region allocation | runtime | `CODE:tt_metal/impl/dispatch/simple_trace_allocator.hpp` | FR-11 |

## F8 — Sub-device & mesh coordination  (item fn §7 bullets: mesh)

| Subfunction | Owner | Key artifacts | Reqs |
|-------------|-------|---------------|------|
| F8.1 Sub-device manager / tracker **[S]** | runtime | `CODE:tt_metal/impl/sub_device/sub_device_manager.hpp`, `CODE:tt_metal/impl/sub_device/sub_device_manager_tracker.hpp` | FR-12, HLR-01e |
| F8.2 Stall-group management **[S]** | runtime | `CODE:tt_metal/api/tt-metalium/device.hpp` (`set_sub_device_stall_group`) | HLR-01e |
| F8.3 Mesh device / submesh / reshape **[S]** | runtime | `CODE:tt_metal/distributed/mesh_device.cpp` | FR-13 |
| F8.4 System mesh over cluster | runtime | `CODE:tt_metal/distributed/system_mesh.cpp` | FR-13 |
| F8.5 Fabric config / control plane **[S]** | runtime→? | `CODE:tt_metal/fabric/control_plane.cpp`, `CODE:tt_metal/fabric/fabric.cpp` | ENV-06 |
| F8.6 Multi-host distributed context | runtime→MPI | `CODE:tt_metal/distributed/multihost/distributed_context.cpp` | FR-14 |

---

## Ownership questions (see runtime-baseline-decisions.md)

- **F8.5 Fabric control plane** lives under `tt_metal/fabric/` but Fabric is listed as a
  separate safety domain — decide whether it is inside metal Runtime or a neighbor.
- **F1.2 / F1.4 device open & topology** straddle metal Runtime and UMD — decide the
  exact contract boundary.
- **F3.3 JIT build → SFPI** — decide tool-qualification ownership (TCL).
- **F6.3–F6.5 fault detection / timeout / safe state** — decide whether reaction /
  safe-state is owned by metal Runtime or a Safety Manager.

## Requirement allocation matrix (summary)

| Function | Safety props (HLR) | Capabilities (FR/PERF/ENV) |
|----------|--------------------|-----------------------------|
| F1 lifecycle | 05b, 05c, 06b | FR-01, ENV-04/05 |
| F2 memory | 01c, 01d, 02a | FR-02, FR-05 |
| F3 build | 02d, 07b | FR-03, FR-04, FR-06 |
| F4 dispatch | 01a, 01b, 01e, 02a, 03c, 06a | FR-07, FR-08 |
| F5 sync | 02b, 03a, 03b | FR-09, FR-10 |
| F6 fault/status | 04a, 04b, 04c, 04d, 05a | FR-15, FR-16, PERF-03 |
| F7 trace | — | FR-11 |
| F8 mesh/subdev | 01e, 07a | FR-12, FR-13, FR-14, ENV-06 |
