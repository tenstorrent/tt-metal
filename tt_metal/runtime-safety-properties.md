# metal Runtime (runtime) — Safety Properties

Negative / invariant requirements: **what must never go wrong**.
Companion to `runtime-item-definition.md` (Section 17) and
`runtime-functional-capabilities.md`.

ASIL target: **ASIL-D** (all requirements below, unless noted).

Conventions:
- Provenance: `SPEC:<id>`, `CODE:<path>`, `HAZARD:§13`, `PRODUCT`.
- Status: `FIRM` (spec-backed), `CANDIDATE` (code-derived, confirm),
  `PROPOSED` (hazard-derived, decide ownership).
- IDs: `runtime-HLR-<nn>` grouped under Safety Goals `SG-<X>`.
- Each requirement lists: shall-statement, ASIL, rationale, allocation, verification.

> No formal specs / product requirements were supplied for this pass; therefore all
> requirements are `CANDIDATE` (code-derived, confirm intent/ownership) or
> `PROPOSED` (hazard-derived gap). None are `FIRM` yet.

Verification methods: Test, Review, Analysis, Walk-through, Fault injection (FI),
Simulation, Static analysis (SA).

---

## SG-A — Targeting / placement integrity

**Goal:** metal Runtime never issues a transfer, allocation, or program launch to an
unintended device, core, sub-device, bank, or memory address.

### runtime-HLR-01a — No out-of-range device target
metal Runtime shall not initiate an operation against a device index that is not part
of the enumerated, reachable device set.
- ASIL: D
- Rationale: §13 "wrong target"; mis-targeting a device is a top-level malfunction.
  Guarded today by `TT_FATAL` on device index bounds
  (`CODE:tt_metal/impl/device/device_manager.cpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, FI (inject invalid device id), Test.
- Provenance: `CODE:tt_metal/impl/device/device_manager.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-01b — No out-of-range core/coordinate target
metal Runtime shall not construct a NoC / core target whose coordinates fall outside
the valid worker/eth/DRAM ranges of the target device.
- ASIL: D
- Rationale: §13 "wrong target" / "wrong interpretation"; validated in core-range
  construction and (opt-in) host NOC sanitize
  (`CODE:tt_metal/common/core_coord.cpp`, `CODE:tt_metal/llrt/sanitize_noc_host.hpp`).
- Allocation: metal Runtime (with device-side Watcher as external check when enabled).
- Verification: Review, SA, FI, Test.
- Provenance: `CODE:tt_metal/common/core_coord.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-01c — No allocation into reserved regions
metal Runtime shall not allocate or write user data into reserved device memory
regions (e.g., addresses below the base allocator address).
- ASIL: D
- Rationale: §13 "wrong target"; enforced by reserved-DRAM `TT_FATAL`
  (`CODE:tt_metal/impl/host_api/tt_metal.cpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, Test.
- Provenance: `CODE:tt_metal/impl/host_api/tt_metal.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-01d — No overlapping live allocations
metal Runtime shall not return two concurrently-live allocations that overlap the
same device memory range.
- ASIL: D
- Rationale: §13 "memory integrity"; overlap corrupts data. Overlap validation exists
  (`CODE:tt_metal/common/core_coord.cpp` — `validate_no_overlap`;
  `CODE:tt_metal/impl/allocator/bank_manager.hpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, FI, Test.
- Provenance: `CODE:tt_metal/impl/allocator/allocator.hpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-01e — Correct sub-device scoping
metal Runtime shall not dispatch work or allocate L1 outside the core set of the
targeted sub-device.
- ASIL: D
- Rationale: §13 "wrong target"; sub-device partitioning isolation
  (`CODE:tt_metal/impl/sub_device/sub_device_manager.hpp`).
- Allocation: metal Runtime.
- Verification: Review, Analysis, Test.
- Provenance: `CODE:tt_metal/impl/sub_device/sub_device_manager.hpp` · `HAZARD:§13` · CANDIDATE

---

## SG-B — Data-movement correctness & integrity

**Goal:** metal Runtime never silently transfers wrong, incomplete, or corrupted data
between host and device.

### runtime-HLR-02a — No size/page/alignment mismatch
metal Runtime shall not perform a buffer transfer when the requested size, page size,
or alignment is inconsistent with the buffer configuration.
- ASIL: D
- Rationale: §13 "commission/wrong value"; guarded by buffer `TT_FATAL` checks
  (`CODE:tt_metal/impl/buffers/buffer.cpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, Test.
- Provenance: `CODE:tt_metal/impl/buffers/buffer.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-02b — No partial transfer reported as complete
metal Runtime shall not report a data transfer as complete unless the full requested
byte range has been committed (write) or read back (read).
- ASIL: D
- Rationale: §13 "commission/omission"; completion is confirmed via completion queue
  (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`).
- Allocation: metal Runtime.
- Verification: Analysis, FI, Test.
- Provenance: `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-02c — Detect corruption of transferred data
metal Runtime shall detect corruption of buffer data transferred between host and
device (e.g., via end-to-end integrity check) and shall not present corrupted data as
valid.
- ASIL: D
- Rationale: §13 "memory integrity"; **no default ECC/CRC/checksum** exists on the
  buffer path today (gap). An optional binary-validation env flag exists
  (`TT_METAL_VALIDATE_PROGRAM_BINARIES`) but is not default and covers binaries only.
- Allocation: metal Runtime and/or lower layer (UMD / hardware ECC) — **to decide**.
- Verification: Analysis, FI, Test.
- Provenance: `HAZARD:§13` · PROPOSED

### runtime-HLR-02d — Detect corruption of loaded program binaries
metal Runtime shall detect corruption or mismatch of a compiled kernel/program binary
before it is executed on device.
- ASIL: D
- Rationale: §13 "memory integrity"; binary integrity is assumed post-compile;
  `TT_METAL_VALIDATE_PROGRAM_BINARIES` is opt-in
  (`CODE:tt_metal/llrt/rtoptions.cpp`).
- Allocation: metal Runtime (JIT build / program subsystem).
- Verification: Analysis, FI, Test, Review.
- Provenance: `CODE:tt_metal/llrt/rtoptions.cpp` · `HAZARD:§13` · PROPOSED

---

## SG-C — Command ordering & synchronization

**Goal:** metal Runtime never violates the ordering / synchronization guarantees the
caller relies on.

### runtime-HLR-03a — No completion signalled before work done
metal Runtime shall not signal an event or return from a blocking synchronization
call before all work it is defined to gate has completed on device.
- ASIL: D
- Rationale: §13 "ordering/timing"; event record uses dispatch barriers + prefetch
  stall (`CODE:tt_metal/impl/event/dispatch.cpp`).
- Allocation: metal Runtime.
- Verification: Analysis, FI, Test.
- Provenance: `CODE:tt_metal/impl/event/dispatch.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-03b — Monotonic, unique event identity
metal Runtime shall not reuse or reorder command-queue event identifiers such that a
later event appears completed before an earlier one on the same queue.
- ASIL: D
- Rationale: §13 "ordering"; monotonic per-CQ event ids
  (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`).
- Allocation: metal Runtime.
- Verification: Analysis, Test.
- Provenance: `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-03c — No command-sequence overrun of dispatch buffers
metal Runtime shall not write a command sequence that exceeds the sized issue/dispatch
buffer region for the target command queue.
- ASIL: D
- Rationale: §13 "commission"; sizes computed before write
  (`CODE:tt_metal/impl/dispatch/device_command_calculator.hpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, Test.
- Provenance: `CODE:tt_metal/impl/dispatch/device_command_calculator.hpp` · `HAZARD:§13` · CANDIDATE

---

## SG-D — Fault detection, propagation & no silent failure

**Goal:** metal Runtime never silently suppresses a detected or detectable device /
dispatch fault.

### runtime-HLR-04a — No silent suppression of lower-layer faults
metal Runtime shall not discard or suppress an error reported by a lower layer (UMD,
device firmware) without propagating it to the caller.
- ASIL: D
- Rationale: §13 "loss of fault propagation"; UMD init-failure action is forced to
  THROW and metal escalates via `TT_THROW`
  (`CODE:tt_metal/impl/debug/watcher_device_reader.cpp`).
- Allocation: metal Runtime.
- Verification: Review, FI, Test.
- Provenance: `CODE:tt_metal/impl/debug/watcher_device_reader.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-04b — Bounded detection of device hang
metal Runtime shall detect a stalled command queue / device hang within a bounded
time and report it rather than waiting indefinitely.
- ASIL: D
- Rationale: §13 "timing"; timeout exists but **default is infinite wait**
  (`timeout_duration_for_operations = 0`,
  `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`,
  `CODE:tt_metal/llrt/rtoptions.hpp`). Default configuration violates this property.
- Allocation: metal Runtime (requires operation timeout enabled and bounded FTTI).
- Verification: FI (induce hang), Test, Analysis.
- Provenance: `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` · `HAZARD:§13` · PROPOSED

### runtime-HLR-04c — Device fault observability available in safety configuration
metal Runtime shall provide device fault detection (NOC / CB / stack / assert /
launch-message sanitization) in the supported safety configuration.
- ASIL: D
- Rationale: §13 "fault-detection availability"; Watcher provides this but is
  **opt-in / non-default** (`CODE:tt_metal/impl/debug/watcher_server.cpp`,
  `CODE:tt_metal/llrt/rtoptions.hpp`). Must be mandated by assumption of use or made
  default for a safety build.
- Allocation: metal Runtime (Watcher) + integration configuration.
- Verification: Review, FI, Test.
- Provenance: `CODE:tt_metal/impl/debug/watcher_server.cpp` · `HAZARD:§13` · PROPOSED

### runtime-HLR-04d — Detected fault does not corrupt other work
metal Runtime shall contain a detected fault so it does not propagate corrupted state
into unrelated concurrent operations on other devices / command queues.
- ASIL: D
- Rationale: §13 "commission" / freedom-from-interference; needs DFA. Today a fault
  raises a host exception; containment across devices is not formally established.
- Allocation: metal Runtime + system-level supervision — **to decide**.
- Verification: Analysis (DFA), FI.
- Provenance: `HAZARD:§13` · PROPOSED

---

## SG-E — Defined safe state on failure

**Goal:** on an unrecoverable fault, metal Runtime leaves the system in a defined
state rather than an undefined or "unrecoverable" one.

### runtime-HLR-05a — Defined state on fault
metal Runtime shall drive the affected device/command-queue to a defined state
(reported error plus known host-side + device-side state) on detection of an
unrecoverable fault.
- ASIL: D
- Rationale: §13 "lifecycle/safe state"; today dispatch timeout labels the device
  "unrecoverable" with **no automatic reset or deterministic safe mode**
  (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`).
- Allocation: metal Runtime and/or external safe-state mechanism — **to decide**.
- Verification: Analysis, FI, Test.
- Provenance: `CODE:tt_metal/impl/dispatch/system_memory_manager.cpp` · `HAZARD:§13` · PROPOSED

### runtime-HLR-05b — No use of a device that failed initialization
metal Runtime shall not service command-queue traffic on a device whose firmware /
dispatch initialization did not complete successfully.
- ASIL: D
- Rationale: §13 "lifecycle"; init failure throws with reset guidance
  (`CODE:tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`) and readiness
  is gated (`CODE:tt_metal/impl/device/device.cpp`).
- Allocation: metal Runtime.
- Verification: Review, FI, Test.
- Provenance: `CODE:tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-05c — Complete, ordered teardown
metal Runtime shall release all host-side dispatch/firmware/CQ/device resources in a
defined order on teardown, including after a partially-completed initialization.
- ASIL: D
- Rationale: §13 "lifecycle"; ordered teardown with `TT_FATAL(init_done_.empty())`
  invariant and partial-init handling
  (`CODE:tt_metal/impl/device/device_manager.cpp`).
- Allocation: metal Runtime.
- Verification: Review, Test (including fault-during-init), Analysis.
- Provenance: `CODE:tt_metal/impl/device/device_manager.cpp` · `HAZARD:§13` · CANDIDATE

---

## SG-F — Configuration & descriptor integrity

**Goal:** metal Runtime never operates on inconsistent or unsupported configuration
without detection.

### runtime-HLR-06a — Reject unsupported command-queue count
metal Runtime shall reject a request for a number of hardware command queues outside
the supported range (1..`MAX_NUM_HW_CQS`).
- ASIL: D
- Rationale: §13 "abstraction mismatch"; `TT_FATAL` on `num_hw_cqs`
  (`CODE:tt_metal/impl/device/device.cpp`; `MAX_NUM_HW_CQS = 2`,
  `CODE:tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp`).
- Allocation: metal Runtime.
- Verification: Review, SA, Test.
- Provenance: `CODE:tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-06b — Reject inconsistent cluster/harvesting descriptor
metal Runtime shall reject or flag a device set whose harvesting / SoC descriptor is
inconsistent across user devices for the intended mode.
- ASIL: D
- Rationale: §13 "config/descriptor faults"; `validate_harvesting_masks`
  (`CODE:tt_metal/llrt/tt_cluster.cpp`).
- Allocation: metal Runtime.
- Verification: Review, Analysis, Test.
- Provenance: `CODE:tt_metal/llrt/tt_cluster.cpp` · `HAZARD:§13` · CANDIDATE

### runtime-HLR-06c — No unsafe configuration silently applied
metal Runtime shall not silently operate in a mode that disables safety-relevant
behavior via configuration (e.g., `TT_METAL_SKIP_LOADING_FW`,
`TT_METAL_DISABLE_DMA_OPS`) without the configuration being part of a controlled,
recorded envelope.
- ASIL: D
- Rationale: §13 "config faults"; many `TT_METAL_*` env vars alter safety-relevant
  behavior (`CODE:tt_metal/llrt/rtoptions.cpp`).
- Allocation: metal Runtime + integration/process control.
- Verification: Review, SA, Analysis.
- Provenance: `CODE:tt_metal/llrt/rtoptions.cpp` · `HAZARD:§13` · PROPOSED

---

## SG-G — Freedom from interference (resource isolation)

**Goal:** independent workloads / command queues / sub-devices do not interfere via
metal Runtime shared state.

### runtime-HLR-07a — No cross-queue state corruption
metal Runtime shall not allow one command queue's dispatch/completion state to corrupt
another queue's state on the same device.
- ASIL: D
- Rationale: §13 "commission"/FFI; per-CQ interfaces + shared CQ state
  (`CODE:tt_metal/impl/dispatch/system_memory_manager.hpp`,
  `CODE:tt_metal/impl/dispatch/cq_shared_state.hpp`). Requires DFA.
- Allocation: metal Runtime.
- Verification: Analysis (DFA), FI, Test.
- Provenance: `CODE:tt_metal/impl/dispatch/cq_shared_state.hpp` · `HAZARD:§13` · PROPOSED

### runtime-HLR-07b — Concurrency-safe shared caches
metal Runtime shall not corrupt shared program / JIT-build caches under concurrent
access from multiple host threads.
- ASIL: D
- Rationale: §13 "commission"/FFI; JIT build cache is described as thread-safe
  (`CODE:tt_metal/jit_build/jit_build_cache.hpp`); program cache concurrency to
  confirm.
- Allocation: metal Runtime.
- Verification: Analysis, SA (TSAN), Test.
- Provenance: `CODE:tt_metal/jit_build/jit_build_cache.hpp` · `HAZARD:§13` · CANDIDATE

---

## Summary

| ID | Safety goal | Status | Allocation note |
|----|-------------|--------|-----------------|
| runtime-HLR-01a..e | SG-A Targeting/placement | CANDIDATE | metal Runtime |
| runtime-HLR-02a..b | SG-B Data movement | CANDIDATE | metal Runtime |
| runtime-HLR-02c..d | SG-B Data/binary integrity | PROPOSED (gap: no ECC/CRC default) | to decide |
| runtime-HLR-03a..c | SG-C Ordering/sync | CANDIDATE | metal Runtime |
| runtime-HLR-04a | SG-D Fault propagation | CANDIDATE | metal Runtime |
| runtime-HLR-04b..d | SG-D Detection/timeout/containment | PROPOSED (gap: opt-in/default-off) | to decide |
| runtime-HLR-05a | SG-E Safe state | PROPOSED (gap: no auto safe state) | to decide |
| runtime-HLR-05b..c | SG-E Init/teardown | CANDIDATE | metal Runtime |
| runtime-HLR-06a..b | SG-F Config integrity | CANDIDATE | metal Runtime |
| runtime-HLR-06c | SG-F Unsafe config | PROPOSED | metal Runtime + process |
| runtime-HLR-07a..b | SG-G Freedom from interference | PROPOSED/CANDIDATE | metal Runtime (needs DFA) |

Gaps requiring human resolution are recorded in `runtime-baseline-decisions.md`.
