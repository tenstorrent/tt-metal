# metal Runtime (runtime) — Functional Capabilities

Positive / measurable requirements: **what the item must do**, plus performance
targets and the supported envelope (scale / topology / configuration limits).
Companion to `runtime-item-definition.md` (Section 17) and
`runtime-safety-properties.md` (authoritative where overlapping).

ASIL target: **ASIL-D**.

Conventions:
- Provenance: `SPEC:<id>`, `CODE:<path>`, `HAZARD:§13`, `PRODUCT`.
- Status: `FIRM`, `CANDIDATE`, `PROPOSED`.
- IDs: capabilities `runtime-FR-<nn>`, performance `runtime-PERF-<nn>`,
  envelope `runtime-ENV-<nn>`.
- Verification: Test, Review, Analysis, Walk-through, Fault injection (FI),
  Simulation, Static analysis (SA), Benchmark.

> No formal specs / product requirements were supplied for this pass. Capabilities
> below are **code-derived (`CANDIDATE`)**; performance and several envelope values are
> `PROPOSED` because no product SLA / supported-config document was supplied. Numeric
> envelope constants that are hard-coded in the source are `CANDIDATE`.

---

## Capability requirements (FR)

### runtime-FR-01 — Device lifecycle management
metal Runtime shall support opening, initializing, and closing devices (single-device
and mesh) into and out of a defined ready state.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/host_api.hpp` (`CreateDevice`/`CloseDevice`),
  `CODE:tt_metal/api/tt-metalium/mesh_device.hpp` (`MeshDevice::create`) · CANDIDATE

### runtime-FR-02 — Device memory allocation
metal Runtime shall support allocation and deallocation of device memory buffers
(interleaved and sharded; L1 and DRAM; single-device and mesh).
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/host_api.hpp` (`CreateBuffer`/`DeallocateBuffer`),
  `CODE:tt_metal/api/tt-metalium/mesh_buffer.hpp` · CANDIDATE

### runtime-FR-03 — Program & kernel creation
metal Runtime shall support creating programs and attaching data-movement / compute
kernels (from file or string) targeted at a core spec.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/host_api.hpp` (`CreateProgram`, `CreateKernel`,
  `CreateKernelFromString`) · CANDIDATE

### runtime-FR-04 — Kernel/program compilation
metal Runtime shall compile programs/kernels to loadable binaries using the RISC-V JIT
toolchain, with a build cache for deduplication.
- ASIL: D · Allocation: metal Runtime (JIT build) + SFPI toolchain (external tool)
- Verification: Test, Review, Analysis
- Provenance: `CODE:tt_metal/jit_build/build.cpp`,
  `CODE:tt_metal/jit_build/jit_build_cache.hpp` · CANDIDATE

### runtime-FR-05 — Circular buffers & semaphores
metal Runtime shall support creating per-core circular buffers and semaphores within a
program.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/host_api.hpp` (`CreateCircularBuffer`,
  `CreateSemaphore`, `CreateGlobalSemaphore`) · CANDIDATE

### runtime-FR-06 — Runtime-argument configuration
metal Runtime shall support setting per-core and common runtime arguments for kernels
prior to dispatch.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/host_api.hpp` (`SetRuntimeArgs`,
  `SetCommonRuntimeArgs`) · CANDIDATE

### runtime-FR-07 — Host↔device data transfer
metal Runtime shall support enqueued host-to-device and device-to-host buffer
transfers (blocking and non-blocking), including mesh buffers and shards.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/mesh_command_queue.hpp`
  (`enqueue_write_mesh_buffer`, `enqueue_read_mesh_buffer`, shards),
  `CODE:tt_metal/api/tt-metalium/tt_metal.hpp` (`detail::WriteToBuffer`,
  `detail::ReadFromBuffer`) · CANDIDATE

### runtime-FR-08 — Workload dispatch
metal Runtime shall support dispatching program workloads to a device / mesh via the
command queue (fast dispatch) and via the slow-dispatch launch path.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/distributed.hpp` (`EnqueueMeshWorkload`),
  `CODE:tt_metal/api/tt-metalium/tt_metal.hpp` (`detail::LaunchProgram`) · CANDIDATE

### runtime-FR-09 — Completion synchronization
metal Runtime shall support blocking completion synchronization for a command queue,
optionally scoped to specified sub-devices.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Analysis
- Provenance: `CODE:tt_metal/api/tt-metalium/distributed.hpp` (`Finish`, `Synchronize`) · CANDIDATE

### runtime-FR-10 — Event record / wait / query
metal Runtime shall support recording events on a command queue and waiting on or
querying those events from host and device.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Analysis
- Provenance: `CODE:tt_metal/api/tt-metalium/mesh_command_queue.hpp`
  (`enqueue_record_event`, `enqueue_wait_for_event`),
  `CODE:tt_metal/api/tt-metalium/distributed.hpp` (`EventSynchronize`, `EventQuery`) · CANDIDATE

### runtime-FR-11 — Trace capture & replay
metal Runtime shall support capturing a command sequence as a trace and replaying it.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/mesh_device.hpp` (`begin_mesh_trace`,
  `end_mesh_trace`, `replay_mesh_trace`),
  `CODE:tt_metal/distributed/mesh_trace.cpp` · CANDIDATE

### runtime-FR-12 — Sub-device management
metal Runtime shall support partitioning a device's worker grid into sub-devices with
independent allocators and stall groups.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/device.hpp`
  (`create_sub_device_manager`, `load_sub_device_manager`, `set_sub_device_stall_group`) · CANDIDATE

### runtime-FR-13 — Mesh / multi-device coordination
metal Runtime shall support coordinating memory, dispatch, and synchronization across a
mesh of devices, including submesh creation and reshape.
- ASIL: D · Allocation: metal Runtime
- Verification: Test, Review
- Provenance: `CODE:tt_metal/api/tt-metalium/mesh_device.hpp` (`shape`, `reshape`,
  `create_submesh`), `CODE:tt_metal/distributed/mesh_device_impl.hpp` · CANDIDATE

### runtime-FR-14 — Multi-host distributed coordination
metal Runtime shall support multi-host coordination (rank/size, barrier, send/recv,
collectives) when built with MPI.
- ASIL: D (to confirm) · Allocation: metal Runtime + MPI environment
- Verification: Test, Analysis
- Provenance: `CODE:tt_metal/api/tt-metalium/distributed_context.hpp`,
  `CODE:tt_metal/distributed/multihost/distributed_context.cpp` · CANDIDATE

### runtime-FR-15 — Fault observability hooks
metal Runtime shall provide device fault observability (Watcher: NOC / CB / stack /
assert / launch-message sanitization) and dispatch operation timeouts as configurable
mechanisms.
- ASIL: D · Allocation: metal Runtime (configuration-gated)
- Verification: Test, FI, Review
- Provenance: `CODE:tt_metal/impl/debug/watcher_server.cpp`,
  `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` · CANDIDATE
- Cross-link: safety-relevant availability is `runtime-HLR-04b/04c` (PROPOSED).

### runtime-FR-16 — Diagnostics / status reporting
metal Runtime shall report operation status and errors to the caller through a defined
path (exception on precondition/fault; boolean on low-level accessors).
- ASIL: D · Allocation: metal Runtime
- Verification: Review, Test
- Provenance: `CODE:tt_stl/tt_stl/assert.hpp`,
  `CODE:tt_metal/api/tt-metalium/host_api.hpp` · CANDIDATE

---

## Performance requirements (PERF)

> No product SLA / performance spec was supplied. The following are placeholders that
> must be quantified from product requirements; verification is benchmark-based, not
> property-based.

### runtime-PERF-01 — Dispatch latency
metal Runtime shall dispatch a program workload to device within `<TARGET_LATENCY>`.
- ASIL: D · Allocation: metal Runtime
- Verification: Benchmark
- Provenance: `PRODUCT` (target not supplied) · PROPOSED

### runtime-PERF-02 — Data-transfer throughput
metal Runtime shall sustain host↔device transfer throughput of `<TARGET_BW>` within
the supported envelope.
- ASIL: D · Allocation: metal Runtime + UMD/DMA
- Verification: Benchmark
- Provenance: `PRODUCT` (target not supplied) · PROPOSED

### runtime-PERF-03 — Fault-detection time (FTTI-relevant)
metal Runtime shall detect and report a device hang / dispatch stall within
`<FTTI_BUDGET>` when operation timeouts are enabled.
- ASIL: D · Allocation: metal Runtime
- Verification: FI, Benchmark
- Provenance: `HAZARD:§13` (FTTI budget not supplied);
  progress-update granularity is configurable
  (`TT_METAL_DISPATCH_PROGRESS_UPDATE_MS`, default 100 ms,
  `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`) · PROPOSED

---

## Supported envelope (ENV)

The envelope defines the scope of "every supported configuration" used by the safety
requirements (e.g., `runtime-HLR-*` are required across this envelope). Values below
are **hard-coded constants / arch parameters observed in source** unless marked
otherwise.

### runtime-ENV-01 — Hardware command queues
Within the supported envelope, metal Runtime shall support 1..2 hardware command
queues per device (`MAX_NUM_HW_CQS = 2`).
- Provenance: `CODE:tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` · CANDIDATE

### runtime-ENV-02 — Dispatch message entries
Within the supported envelope, dispatch supports up to
`DISPATCH_MAX_MESSAGE_ENTRIES = 8` message entries.
- Provenance: `CODE:tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` · CANDIDATE

### runtime-ENV-03 — Host memory channels
Within the supported envelope, metal Runtime supports up to 4 host memory channels per
MMIO device (`MAX_HOST_MEM_CHANNELS = 4`) and asserts ≤ 9 devices per MMIO gateway
group.
- Provenance: `CODE:tt_metal/third_party/umd/device/hugepage.hpp`,
  `CODE:tt_metal/llrt/tt_cluster.cpp` · CANDIDATE

### runtime-ENV-04 — Supported architectures
Within the supported envelope, metal Runtime supports the architectures for which a
HAL and SoC descriptor exist: Wormhole B0, Blackhole, Quasar.
- Provenance: `CODE:tt_metal/llrt/tt_cluster.cpp` (SoC YAML selection),
  `CODE:tt_metal/llrt/hal/` · CANDIDATE
- Note: which arch(es) are in the *safety* baseline is an open decision
  (`runtime-baseline-decisions.md`).

### runtime-ENV-05 — Per-arch memory parameters
Within the supported envelope, metal Runtime operates within the per-arch memory
parameters below (from SoC descriptors / UMD arch headers):

| Arch | Worker L1 | DRAM bank size | ETH L1 | DRAM banks |
|------|-----------|----------------|--------|-----------|
| Wormhole B0 | 1,499,136 B (~1.43 MB) | 2,147,483,648 B (2 GB) | 262,144 B | 6 |
| Blackhole | 1,572,864 B (1.5 MB) | 4,278,190,080 B (~4 GB) | 524,288 B | 8 |
| Quasar | 4,194,304 B (4 MB) | 1,073,741,824 B (1 GB) | 0 | 2 |

- Provenance: `CODE:tt_metal/soc_descriptors/*.yaml`,
  `CODE:tt_metal/third_party/umd/device/api/.../*_implementation.hpp` · CANDIDATE

### runtime-ENV-06 — Mesh / cluster topologies
Within the supported envelope, metal Runtime supports the cluster/mesh topologies
classified by the runtime (e.g., N150, N300_2x2, T3K, P100/P150_X{2,4,8}, P300{_X2},
TG, GALAXY, BLACKHOLE_GALAXY; CUSTOM requires a custom mesh graph descriptor). Mesh
shapes are N-D via `MeshShape` bounded by discovered chip count.
- Provenance: `CODE:tt_metal/llrt/tt_cluster.cpp` (cluster-type classification),
  `CODE:tt_metal/api/tt-metalium/mesh_coord.hpp` (`MeshShape`) · CANDIDATE
- Note: which topologies are in the safety baseline is an open decision.

### runtime-ENV-07 — Dispatch mode
Within the supported envelope, metal Runtime operates in fast-dispatch mode by
default; slow-dispatch mode is available (`TT_METAL_SLOW_DISPATCH_MODE`) and is forced
in emule mode.
- Provenance: `CODE:tt_metal/llrt/rtoptions.cpp` · CANDIDATE
- Note: which dispatch mode is in the safety baseline is an open decision.

### runtime-ENV-08 — DMA constraints
Within the supported envelope, metal Runtime host DMA is available on Wormhole via
MMIO for transfers ≥ 32 bytes; Blackhole DMA is not yet supported on the metal path.
- Provenance: `CODE:tt_metal/llrt/tt_cluster.cpp` · CANDIDATE

### runtime-ENV-09 — Backend targets
Within the supported envelope, metal Runtime supports Silicon, Simulation, Mock, and
SW-Emule backends selected by environment; the safety baseline target backend is
Silicon (to confirm).
- Provenance: `CODE:tt_metal/llrt/rtoptions.cpp`,
  `CODE:tt_metal/third_party/umd/device/api/umd/device/cluster.hpp` (`ChipType`) · CANDIDATE

---

## Traceability note

Every FR above maps to one or more functions in `runtime-functions.md` and, where it
concerns a negative/invariant property, cross-links to a `runtime-HLR-*` in
`runtime-safety-properties.md`. Envelope requirements bound the "every supported
configuration" scope of the safety properties.
