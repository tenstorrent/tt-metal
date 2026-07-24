# metal Runtime (runtime) — Assumptions, Preconditions, Postconditions, Dependencies

Companion record to `runtime-item-definition.md` (§9–12). Variant-generic content
stays here; variant-specific assumptions are tagged **[VARIANT]** and must be
specialized per arch / cluster / dispatch mode during tailoring.

Each assumption/precondition is tagged with an enforceability note:
- **[ENF]** enforced in code today (with `CODE:` reference)
- **[CHK]** partially checked / opt-in
- **[UNENF]** assumed, not enforced (must be covered by product safety concept or
  integration verification)

---

## A. Assumptions of use

### A.1 Caller behavior

- **AoU-01 [UNENF]** Higher-level software uses supported metal Runtime interfaces
  correctly: valid handles, valid device/core/address ranges, correct call ordering
  (create → configure → enqueue → synchronize → close).
- **AoU-02 [UNENF]** A given command queue is used single-threaded, or accesses are
  externally synchronized by the caller. (Thread-local CQ id APIs exist:
  `CODE:tt_metal/api/tt-metalium/host_api.hpp`.)
- **AoU-03 [UNENF]** Host buffers passed to enqueue read/write remain valid and
  unmodified for the duration of a non-blocking transfer until completion is
  confirmed.
- **AoU-04 [UNENF]** Kernel source / runtime args supplied by the caller are correct;
  metal Runtime validates structure, not functional correctness of user kernels.

### A.2 Lower-layer / platform behavior

- **AoU-05 [UNENF]** **UMD** is present, of a compatible version, and performs
  physical device access (PCIe/DMA, register R/W, reset, sysmem) per its own spec
  (`CODE:tt_metal/third_party/umd/device/api/umd/device/cluster.hpp`).
- **AoU-06 [ENF]** The **HAL** for the target arch is available and correctly
  describes the memory map / addressing (`CODE:tt_metal/llrt/hal.cpp`).
- **AoU-07 [UNENF]** The **RISC-V JIT toolchain (SFPI, `riscv-tt-elf-g++`)** in the
  environment is the qualified/intended version and produces correct binaries
  (`CODE:tt_metal/jit_build/build.cpp`). Tool-qualification open item.
- **AoU-08 [UNENF]** Underlying device, NoC, Fabric, and firmware hardware behave per
  their own specifications.

### A.3 Configuration / environment

- **AoU-09 [CHK]** Cluster/SoC descriptor, harvesting, and connectivity information
  presented to metal Runtime are valid for the intended mode and consistent across
  user devices (`CODE:tt_metal/llrt/tt_cluster.cpp` — `validate_harvesting_masks`).
- **AoU-10 [UNENF]** `TT_METAL_*` environment configuration is set to values within
  the supported safety envelope; debug/observability toggles that weaken safety are
  controlled (`CODE:tt_metal/llrt/rtoptions.cpp`).
- **AoU-11 [UNENF] [SAFETY-CRITICAL]** For a safety configuration, device fault
  observability (**Watcher**) and **operation timeouts** are enabled and configured,
  since they are opt-in / default-off in code
  (`CODE:tt_metal/impl/debug/watcher_server.cpp`,
  `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`).
- **AoU-12 [UNENF]** metal Runtime is built in a configuration where safety-relevant
  invariant checks are active (note: `TT_ASSERT` is stripped in release builds;
  `TT_FATAL`/`TT_THROW` remain).

### A.4 System-level safety

- **AoU-13 [UNENF]** System-level supervision and external safety mechanisms (Safety
  Manager, watchdog, reset/safe-state actuation, FTTI arbitration) exist **outside**
  the metal Runtime boundary where required by the product safety concept — because
  metal Runtime has no automatic safe-state/reset today.

### A.5 Variant-specific assumptions [VARIANT]

- **AoU-V1 [VARIANT]** Target architecture is one of Wormhole B0 / Blackhole / Quasar
  with matching HAL and SoC descriptor (`CODE:tt_metal/llrt/tt_cluster.cpp`).
- **AoU-V2 [VARIANT]** Cluster/mesh topology is one of the classified supported types
  (see `runtime-ENV-06`); CUSTOM topologies require a validated mesh graph descriptor.
- **AoU-V3 [VARIANT]** Dispatch mode (fast vs slow) is the one qualified for the
  variant; emule/simulator/mock backends are not the silicon safety target.
- **AoU-V4 [VARIANT]** DMA usage respects arch constraints (WH MMIO ≥ 32 B; BH DMA
  not on metal path) (`CODE:tt_metal/llrt/tt_cluster.cpp`).
- **AoU-V5 [VARIANT]** Fabric configuration and reliability mode match the physical
  system for multi-chip variants (`CODE:tt_metal/fabric/fabric.cpp`).

---

## B. Preconditions (before a supported operation)

- **PRE-01 [ENF]** `RunTimeOptions` / `MetalContext` initialized
  (`CODE:tt_metal/impl/context/metal_env.cpp`).
- **PRE-02 [ENF]** UMD driver started and cluster discovered
  (`CODE:tt_metal/llrt/tt_cluster.cpp`).
- **PRE-03 [ENF]** HAL constructed for the discovered arch (`CODE:tt_metal/llrt/hal.cpp`).
- **PRE-04 [ENF]** Target device(s) enumerated and reachable
  (`CODE:tt_metal/impl/device/device_manager.cpp`).
- **PRE-05 [ENF]** Firmware / RISC and dispatch-kernel init completed successfully
  (`CODE:tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`,
  `CODE:tt_metal/impl/dispatch/topology.hpp`).
- **PRE-06 [ENF]** Device reached ready state before servicing CQ traffic; fast
  dispatch requires CQ kernels compiled/configured
  (`CODE:tt_metal/impl/device/device.cpp`).
- **PRE-07 [CHK]** Barrier addresses, host-memory channels, and (Blackhole) NOC
  translation prerequisites satisfied (`CODE:tt_metal/llrt/tt_cluster.cpp`).
- **PRE-08 [ENF]** For an enqueue operation: device open, program compiled (fast
  dispatch compiles lazily on first `EnqueueMeshWorkload`), buffers allocated, CQ id
  valid (`CODE:tt_metal/impl/device/device.cpp`).

---

## C. Postconditions

### C.1 On success (operation scope = single enqueued command up to next sync, or full lifecycle)

- **POST-01** Intended transfer / launch / event issued correctly to the intended
  target within the boundary (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`).
- **POST-02** Completion / status / error available to caller via defined path
  (`Finish` / event query / return value) (`CODE:tt_metal/api/tt-metalium/distributed.hpp`).
- **POST-03** Monitoring / diagnostic outputs updated where enabled (Watcher log,
  Inspector, telemetry).
- **POST-04** Host-side staging contents are transient / not reliably inspectable.
- **POST-05** On teardown, host-side dispatch/firmware/CQ/device resources released in
  defined order; resources reusable for future launches
  (`CODE:tt_metal/impl/device/device_manager.cpp`).

### C.2 On failure

- **POST-F1** A precondition violation or detected fault produces a reported error:
  `TT_FATAL`/`TT_THROW` → `std::runtime_error`, or `bool false` on low-level accessors
  (`CODE:tt_stl/tt_stl/assert.hpp`).
- **POST-F2** A device NOC/assert fault (Watcher enabled) results in a device soft-hang
  spin loop plus host exception (`CODE:tt_metal/impl/debug/watcher_device_reader.cpp`).
- **POST-F3 [GAP]** On dispatch timeout the device is labeled **"unrecoverable"** with
  **no automatic reset / deterministic safe state**
  (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`). Safe-state achievement is
  currently allocated to external mechanisms (AoU-13) — open decision.
- **POST-F4** After a caught exception, host-side context can be torn down via the
  ordered teardown path; device silicon is **not** reset by metal Runtime.

---

## D. Dependencies

Safety-relevant marked **[S]**.

| ID | Dependency | Type | [S] | Reference |
|----|------------|------|-----|-----------|
| DEP-01 | Higher SW consumers (correct usage) | SW (up) | [S] | AoU-01..04 |
| DEP-02 | UMD (device access / reset / sysmem / topology) | SW (down) | [S] | `CODE:tt_metal/third_party/umd` |
| DEP-03 | HAL (arch memory map / addressing) | SW (down) | [S] | `CODE:tt_metal/llrt/hal.hpp` |
| DEP-04 | SFPI RISC-V JIT toolchain | Tool | [S] | `CODE:tt_metal/jit_build/build.cpp` |
| DEP-05 | Cluster/SoC descriptor, coord/harvesting data | Data/Env | [S] | `CODE:tt_metal/llrt/tt_cluster.cpp` |
| DEP-06 | Device / NoC / Fabric interconnect HW | HW/Env | [S] | environmental |
| DEP-07 | RunTimeOptions / `TT_METAL_*` env | Config/Env | [S] | `CODE:tt_metal/llrt/rtoptions.cpp` |
| DEP-08 | MPI runtime (OPEN_MPI) for multi-host | SW/Env | (variant) | `CODE:tt_metal/distributed/multihost/distributed_context.cpp` |
| DEP-09 | Reset / interrupt / memory-barrier / fault-report mechanisms | HW/SW | [S] | `CODE:tt_metal/llrt/tt_cluster.cpp` |
| DEP-10 | Fabric control plane / mesh graph descriptor | SW/Env | [S] | `CODE:tt_metal/fabric/control_plane.cpp` |

Dependencies DEP-02..DEP-06 and DEP-09 cross the item boundary and are the primary
subjects for Dependent Failure Analysis (DFA) and interface safety requirements.
