# metal Runtime (runtime) — Preliminary Failure / Malfunction Considerations

Preliminary malfunction list for metal Runtime, grouped by failure theme using
ISO 26262-style malfunction language (wrong / unintended / loss of / too early / too
late). **No ASILs are assigned here** — those come from HARA. Each malfunction maps to
the affected function(s) in `runtime-functions.md` and to the requirement(s) in
`runtime-safety-properties.md` (HLR) / `runtime-functional-capabilities.md`.

Columns: ID · Malfunction · Function(s) · Requirement(s) · Existing mitigation
(code) · Notes / gap.

---

## T1 — Wrong target / placement

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T1-01 | Operation issued to wrong / out-of-range device | F1.4, F4.2 | HLR-01a | `TT_FATAL` device index bounds (`CODE:tt_metal/impl/device/device_manager.cpp`) | Fatal-throw only |
| FM-T1-02 | Command targets wrong / invalid core coordinate | F4.2, F6.6 | HLR-01b | core-range validation (`CODE:tt_metal/common/core_coord.cpp`); host NOC sanitize (opt-in) | Sanitize opt-in |
| FM-T1-03 | Buffer allocated into reserved region | F2.4 | HLR-01c | reserved-DRAM `TT_FATAL` (`CODE:tt_metal/impl/host_api/tt_metal.cpp`) | — |
| FM-T1-04 | Two live allocations overlap | F2.1, F2.5 | HLR-01d | overlap validation (`CODE:tt_metal/common/core_coord.cpp`), bank free-list | Confirm all paths |
| FM-T1-05 | Work dispatched outside targeted sub-device | F8.1, F8.2 | HLR-01e | sub-device manager + stall groups (`CODE:tt_metal/impl/sub_device/sub_device_manager.hpp`) | Needs isolation analysis |
| FM-T1-06 | Wrong virtual↔physical coordinate translation | F4.2 | HLR-01b | coord map (`CODE:tt_metal/llrt/tt_cluster.cpp`) | Harvesting-dependent |

## T2 — Wrong value / commission (corrupted or incorrect data/command)

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T2-01 | Transfer with size/page/alignment mismatch | F2.3, F4.10 | HLR-02a | buffer `TT_FATAL` (`CODE:tt_metal/impl/buffers/buffer.cpp`) | — |
| FM-T2-02 | Corrupted buffer data undetected | F4.10 | HLR-02c | none by default (no ECC/CRC) | **GAP** |
| FM-T2-03 | Corrupted / mismatched program binary executed | F3.2, F3.8 | HLR-02d | `TT_METAL_VALIDATE_PROGRAM_BINARIES` (opt-in) (`CODE:tt_metal/llrt/rtoptions.cpp`) | **GAP** (opt-in) |
| FM-T2-04 | Wrong runtime args written | F3.10 | HLR-02a | `TT_ASSERT` (debug-only) (`CODE:tt_metal/api/tt-metalium/runtime_args_data.hpp`) | Release strips assert |
| FM-T2-05 | Command sequence overruns dispatch buffer | F4.3, F4.4 | HLR-03c | size calculator before write (`CODE:tt_metal/impl/dispatch/device_command_calculator.hpp`) | — |
| FM-T2-06 | Wrong dispatch-core placement | F4.8 | FR-08 | dispatch_core_manager placement | — |

## T3 — Omission (loss of function)

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T3-01 | Required transfer / launch not issued | F4.5, F4.6 | HLR-02b | completion confirmation via CQ | — |
| FM-T3-02 | Completion never signalled (lost completion) | F5.1 | HLR-02b, HLR-04b | completion-queue polling; timeout (opt-in) | **GAP** (default infinite wait) |
| FM-T3-03 | Event never recorded / lost | F5.2, F5.3 | HLR-03a | dispatch barrier + prefetch stall (`CODE:tt_metal/impl/event/dispatch.cpp`) | — |
| FM-T3-04 | Fault from lower layer dropped (not propagated) | F6.2 | HLR-04a | `TT_THROW` escalation; UMD init action THROW | — |

## T4 — Ordering

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T4-01 | Completion signalled before work done | F5.1, F5.2 | HLR-03a | barrier/stall in event record | — |
| FM-T4-02 | Events reordered / id reused | F5.3 | HLR-03b | monotonic per-CQ event ids (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`) | — |
| FM-T4-03 | Launch message read pointer inconsistency | F4.7, F6.7 | HLR-03a | launch-msg validation in Watcher (`CODE:tt_metal/impl/debug/watcher_device_reader.cpp`) | Watcher opt-in |
| FM-T4-04 | Missing barrier/stall between dependent ops | F4.7, F5.2 | HLR-03a | dispatch WAIT commands | Needs analysis |

## T5 — Timing / hang

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T5-01 | Command queue / device hangs, host waits forever | F5.1, F6.3 | HLR-04b, PERF-03 | `TT_METAL_OPERATION_TIMEOUT_SECONDS` (opt-in; default 0 = infinite) (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`) | **GAP** (default-off) |
| FM-T5-02 | Fetch-queue stall not detected | F6.3 | HLR-04b | fetch-queue timeout THROW (opt-in) | **GAP** (default-off) |
| FM-T5-03 | Operation completes too late (SLA/FTTI miss) | F4–F5 | PERF-01, PERF-03 | none quantified | **GAP** (no FTTI budget) |
| FM-T5-04 | Firmware init does not complete | F1.5 | HLR-05b | bounded FW init wait + THROW (`CODE:tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`) | — |

## T6 — Status reporting / silent failure

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T6-01 | Fault detectable but detection disabled | F6.5 | HLR-04c | Watcher (opt-in) (`CODE:tt_metal/impl/debug/watcher_server.cpp`) | **GAP** (default-off) |
| FM-T6-02 | Device fault silently suppressed | F6.2, F6.5 | HLR-04a, HLR-04c | Watcher `TT_THROW` when enabled | **GAP** if Watcher off |
| FM-T6-03 | Wrong completion/error status returned | F5.1, F6.1 | HLR-02b | `bool` returns / exceptions | Confirm all `bool` paths checked |
| FM-T6-04 | `TT_ASSERT` checks compiled out in release | F6.1 | HLR-02a | debug-only asserts (`CODE:tt_stl/tt_stl/assert.hpp`) | **GAP** (release build) |

## T7 — Memory / freedom-from-interference

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T7-01 | Cross-command-queue state corruption | F4.6, F5.1 | HLR-07a | per-CQ interfaces; shared state (`CODE:tt_metal/impl/dispatch/cq_shared_state.hpp`) | Needs DFA |
| FM-T7-02 | Shared program/JIT cache corrupted under concurrency | F3.4, F3.7 | HLR-07b | thread-safe JIT cache (`CODE:tt_metal/jit_build/jit_build_cache.hpp`) | Confirm program cache |
| FM-T7-03 | Use-after-free of buffer | F2.3 | HLR-01d | emule ASAN (opt-in build) | **GAP** (opt-in) |
| FM-T7-04 | Device-side stack overflow undetected | F6.5 | HLR-04c | Watcher stack-usage check (opt-in) | **GAP** (default-off) |

## T8 — Abstraction / configuration mismatch

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T8-01 | Unsupported num_hw_cqs requested | F4.1 | HLR-06a | `TT_FATAL` (`MAX_NUM_HW_CQS=2`) (`CODE:tt_metal/impl/device/device.cpp`) | — |
| FM-T8-02 | Harvesting inconsistent across devices | F1.2 | HLR-06b | `validate_harvesting_masks` (`CODE:tt_metal/llrt/tt_cluster.cpp`) | — |
| FM-T8-03 | Unsupported mesh shape / topology | F8.3, F8.5 | ENV-06 | topology mapper constraints; `reshape` throws (`CODE:tt_metal/distributed/mesh_device.cpp`) | Confirm coverage |
| FM-T8-04 | Unsafe env-var configuration silently applied | F1.1 | HLR-06c | none (env is authoritative) (`CODE:tt_metal/llrt/rtoptions.cpp`) | **GAP** |
| FM-T8-05 | Stale/incorrect SoC/cluster descriptor | F1.2 | HLR-06b | descriptor validation partial | Confirm |
| FM-T8-06 | Blackhole NOC translation not enabled | F1.2 | HLR-06b | `TT_FATAL` gate (`CODE:tt_metal/llrt/tt_cluster.cpp`) | — |

## T9 — Lifecycle / safe state

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T9-01 | Device used after failed init | F1.7 | HLR-05b | readiness gating (`CODE:tt_metal/impl/device/device.cpp`) | — |
| FM-T9-02 | Incomplete teardown after partial init | F1.8 | HLR-05c | ordered teardown + `init_done_` invariant (`CODE:tt_metal/impl/device/device_manager.cpp`) | — |
| FM-T9-03 | No defined safe state on unrecoverable fault | F6.4 | HLR-05a | device labeled "unrecoverable"; no auto reset (`CODE:tt_metal/impl/dispatch/system_memory_manager.cpp`) | **GAP** |
| FM-T9-04 | Fault not contained to affected device/CQ | F6.4 | HLR-04d | none formal | **GAP** (needs DFA) |

## T10 — Performance

| ID | Malfunction | Fn | Req | Existing mitigation | Notes |
|----|-------------|----|-----|---------------------|-------|
| FM-T10-01 | Dispatch latency exceeds SLA | F4 | PERF-01 | none quantified | **GAP** (no SLA) |
| FM-T10-02 | Transfer throughput below SLA | F4.10 | PERF-02 | none quantified | **GAP** (no SLA) |

---

## Gap summary (feeds runtime-baseline-decisions.md)

The following malfunctions currently have **no default mitigation** or rely on
**opt-in** mechanisms and are the primary safety gaps:

- Data / binary integrity: FM-T2-02, FM-T2-03 (no default ECC/CRC/hash).
- Hang / timeout: FM-T3-02, FM-T5-01, FM-T5-02 (timeout default-off).
- Fault detection availability: FM-T6-01, FM-T6-02, FM-T7-04 (Watcher default-off).
- Release-build assertion stripping: FM-T6-04.
- Safe state / containment: FM-T9-03, FM-T9-04 (no auto safe state; no DFA).
- Unsafe configuration: FM-T8-04.
- Performance / FTTI budgets: FM-T5-03, FM-T10-01, FM-T10-02 (no product SLA/FTTI).
