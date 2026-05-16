<!--
SUMMARY: Adversarial v3 audit of EDM/ETH/ERISC firmware for deadlocks, races, missed signals, hang vectors
KEYWORDS: edm, fabric, erisc, hang, race-condition, wormhole, blackhole, firmware, spin-loop, termination
SOURCE: Manual serial audit of tt-metal branch nsexton/0-racecondition-hunt (10 core files)
SCOPE: All spin loops, shared-memory accesses, ETH TX queue handling, flow control, handshake, teardown
USE WHEN: Investigating ETH channel hangs, ERISC lockups, fabric teardown failures, or credit starvation
-->

# EDM Fabric Firmware Hang-Tracing Audit v3

**Branch**: `nsexton/0-racecondition-hunt`
**Date**: 2026-05-10
**Auditor**: BrAIn (automated adversarial audit, Opus model)

## Methodology

For each of 10 files, audited serially:
- Every unbounded loop: exit conditions, setters, timeout, recovery
- Every shared memory location: cache invalidation, store barriers, torn writes
- Every ETH send: TXQ busy behavior, backpressure, error handling
- Every shared flag: atomicity and reset safety

**Already applied fixes verified present**: FIX HS1, FIX HS2, FIX ER1, FIX AH

---

## Summary of Findings

```
ID        Severity  File                                           Line(s)    Status
WFEWS     MEDIUM    edm_fabric_worker_adapters.hpp                 304-307    UNBOUNDED CREDIT WAIT
MUX-NOWH  MEDIUM    tt_fabric_mux.cpp                              52-56      NO TERM CHECK AT ALL
WH-TERM   INFO      (multiple files)                               --         KNOWN ARCH DECISION
ER3       LOW       fabric_erisc_router_speedy_path.hpp            236        HIDDEN GLOBAL VARIABLE
ER4       LOW       fabric_erisc_router_speedy_path.hpp            179        POTENTIAL UNDERFLOW
FC1       LOW       fabric_router_flow_control.hpp                 247-248    UNBOUNDED TXQ SPIN (hot path avoids)
PT1       LOW       fabric_edm_packet_transmission.hpp             116-130    UNBOUNDED TRID FLUSH
RY1       LOW       fabric_router_relay_extension.cpp              458-478    RESPONSE POOL STALL
EX1       INFO      erisc.cc                                       116-167    STALE GO_MSG RECOVERY
ER2       INFO      fabric_erisc_router.cpp                        2862-2878  INTENTIONAL NO-TERM BARRIER
```

**CRITICAL findings**: 0
**HIGH findings**: 0
**MEDIUM findings**: 2
**LOW findings**: 5
**INFORMATIONAL findings**: 3
**CLEAN verifications**: 21

---

## Detailed Findings

### WFEWS [MEDIUM] — `wait_for_empty_write_slot()` Unbounded With No Termination Check

**File**: `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:304-308`

```cpp
FORCE_INLINE void wait_for_empty_write_slot() const {
    WAYPOINT("FWSW");
    while (!this->edm_has_space_for_packet<1>());
    WAYPOINT("FWSD");
}
```

**Hazard**: Called from `forward_data()` in both mux kernels (legacy and extension) when a packet is ready to send. If the downstream EDM never frees a write slot (e.g., ETH link down, ERISC hang, credit deadlock), this spins forever. The outer main loop's termination check is never reached because control is stuck inside this function.

**Callers**:
- `fabric_router_mux_extension.cpp` — inside `forward_data()` hot path
- `tt_fabric_mux.cpp:125` — inside `forward_data()` hot path
- Multiple UDM paths in `tt_fabric_udm.hpp`

**Impact**: A stalled downstream EDM wedges the entire mux kernel, preventing relay and worker drain, causing cascading hangs. Host teardown writes `TerminationSignal` but the mux never reads it.

**Mitigating factors**:
- Downstream EDM is expected to always make progress
- If downstream is truly hung, adding termination check alone doesn't fix the system
- Performance-sensitive hot path — adding checks has measurable impact

**Verdict**: MEDIUM — Real unbounded spin, but architectural (assumes downstream makes progress). Needs careful perf benchmarking before fixing.

---

### MUX-NOWH [MEDIUM] — Legacy MUX `wait_for_static_connection_to_ready` Has NO Termination Check

**File**: `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:52-56`

```cpp
template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }
    worker_interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE>();
}
```

**Hazard**: Unlike the relay/mux-extension variants (which have `#ifndef ARCH_WORMHOLE` guard), this has NO termination check on ANY architecture. However, examining callers — this function is NOT called from `kernel_main()` in the legacy mux. The legacy mux does not use persistent channels.

**Verdict**: MEDIUM — Dead code path (defined but not called). Cleanup candidate, not urgent.

---

### ER3 [LOW] — Speedy path `did_something` references undeclared variable

**File**: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp:236`

**Description**: Line 236 sets `did_something = true;` but `did_something` is not declared in `run_receiver_channel_step_speedy()`. It must be a global/file-scope variable from `fabric_erisc_router.cpp`. This creates a hidden coupling — the speedy path receiver function silently modifies a variable from the caller's scope. The function already returns a `progress` boolean that serves the same purpose.

**Impact**: Functionally benign (optimization hint) but fragile if variable semantics change.

---

### ER4 [LOW] — Speedy sender `sender_amort_counter` potential underflow

**File**: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp:179`

```cpp
sender_state.sender_amort_counter -= sender_state.completion_count;
```

**Description**: Both fields are unsigned. If `completion_count > sender_amort_counter` (possible if completions arrive faster than sends in a burst), the subtraction underflows to a huge value, making `check_completions` always true. Self-correcting once more sends occur, but creates unnecessary work in the interim.

---

### FC1 [LOW] — `receiver_send_completion_ack<CHECK_BUSY=true>` unbounded ETH TXQ spin

**File**: `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp:247-248`

**Description**: When `CHECK_BUSY=true`, spins on `eth_txq_is_busy(receiver_txq_id)` with empty loop body. All speedy-path and most normal-path call sites use `CHECK_BUSY=false`. Hardware is expected to drain committed TX operations.

---

### PT1 [LOW] — `flush_write_to_noc_pipeline()` unbounded TRID flush loops

**File**: `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:116-130`

**Description**: Iterates over all TRIDs per channel, spinning on `ncrisc_noc_nonposted_write_with_transaction_id_flushed()` for each. When `local_chip_noc_equals_downstream_noc` is false, two spins per TRID. Called from packet dispatch when `flush` flag is set. NOC write completion is a hardware guarantee.

---

### RY1 [LOW] — Relay response pool can stall under persistent mux back-pressure

**File**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_relay_extension.cpp:458-478`

**Description**: `process_response()` sends one response per call. If mux is full, response stays in pool. If pool fills AND mux stays full, relay cannot accept new packets (ASSERT fires). Broken by outer termination check loop. Capacity planning concern, not code bug.

---

### WH-TERM [INFO] — Wormhole Termination Check Removal Pattern

**Locations**: Multiple files — `fabric_erisc_router.cpp:2721`, `fabric_router_relay_extension.cpp:145`, `fabric_router_mux_extension.cpp:120`, `fabric_router_eth_handshake.hpp:34,77`

**Description**: `#ifndef ARCH_WORMHOLE` guards remove `got_immediate_termination_signal()` checks from init wait loops. On WH these loops are truly unbounded. Deliberate — WH uses different termination mechanism (routing FW layer).

---

### ER2 [INFO] — `wait_for_other_local_erisc()` intentionally no termination check

**File**: `fabric_erisc_router.cpp:2862-2878`

**Description**: Two-ERISC barrier sync. Comment explicitly states no termination check is intentional — breaking out early would leave other ERISC spinning forever.

---

### EX1 [INFO] — `erisc.cc` stale go_msg recovery relies on host timeout

**File**: `tt_metal/hw/firmware/src/tt-1xx/erisc.cc:116-167`

**Description**: If kernel crashes without setting `RUN_MSG_DONE`, ERISC spins waiting for next go signal. Host 500ms timeout path handles this (documented extensively in AI journal — FIX PD, PE, PF).

---

## Clean Verifications (No Hazards Found)

### File 1: `fabric_erisc_router.cpp` (3822 lines)
1. **Main loop (line 2291)**: `while (continue_running_main_run_loop<...>(...))` — Has termination check. CLEAN.
2. **Post-send TXQ drain (line 622)**: `while (internal_::eth_txq_is_busy(sender_txq_id)) {}` — Packet is committed; hardware drain guaranteed. Post-send NO teardown exit is correct (would corrupt destination L1). CLEAN.
3. **Context switch coordination (lines 1442, 1459, 1486, 1494)**: All have termination signal checks. CLEAN.
4. **Teardown sequence**: 3x `wait_for_other_local_erisc()`, noc barriers, TERMINATED write. Correct ordering. CLEAN.
5. **Speedy path sender (speedy_path.hpp:82-183)**: Pre-send teardown check at line 124. Post-send TXQ drain at line 141-149 (hardware guaranteed, no teardown — correct). CLEAN.
6. **Speedy path receiver (speedy_path.hpp:203-283)**: Ping-pong TRID with FIX ER1. Single-TRID flush per batch. CLEAN.

### File 2: `fabric_router_relay_extension.cpp` (619 lines)
7. **Main loop (line 598)**: `while (!got_immediate_termination_signal<true>(...))` — CLEAN.
8. **`wait_for_mux_endpoint_ready` (lines 156-184)**: Bounded to 1M iterations, falls through on timeout. CLEAN.
9. **Local mux status wait (line 559)**: Bounded 1M iterations with BH termination check. CLEAN.
10. **Teardown (lines 615-618)**: `noc_async_write_barrier()` + `noc_async_atomic_barrier()` then TERMINATED. CLEAN.

### File 3: `fabric_router_mux_extension.cpp` (401 lines)
11. **Main loop (line 365)**: `while (!got_immediate_termination_signal<true>(...))` — CLEAN.
12. **`wait_for_fabric_endpoint_ready` (line 333)**: Bounded, returns false on timeout, caller handles. CLEAN.
13. **Persistent channel waits (lines 348-358)**: Calls relay-style `wait_for_static_connection_to_ready` (with WH guard). CLEAN.

### File 4: `fabric_erisc_datamover_channels.hpp` (583 lines)
14. Pure data structures and buffer management. No loops, no waits. CLEAN.

### File 5: `fabric_edm_packet_transmission.hpp` (488 lines)
15. Packet dispatch/forwarding with TRID-based writes. No polling beyond PT1. CLEAN.

### File 6: `fabric_router_flow_control.hpp` (265 lines)
16. Pure credit management data structures. ETH TXQ spins noted in FC1. CLEAN.

### File 7: `edm_handshake.hpp` (253 lines)
17. **FIX HS1 verified** at line 130. **FIX AH verified** at lines 60-76 (guarded TXQ flush). CLEAN.
18. **Deprecated split handshake** (lines 157-245): `sender_side_start` / `receiver_side_finish` — legacy code, bounded by `eth_wait_receiver_done` / `eth_wait_for_bytes`. CLEAN.

### File 8: `fabric_router_eth_handshake.hpp` (104 lines)
19. **FIX HS2 verified** at lines 50-61. WH-TERM pattern noted. CLEAN.

### File 9: `tt_fabric_mux.hpp` + `tt_fabric_mux.cpp` (40 + 308 lines)
20. **Main loop (line 241)**: Termination check + graceful drain (lines 242-256). CLEAN.
21. **Teardown ordering (lines 286-306)**: `close_start()` → barriers → TERMINATED → `close_finish()`. Comment explains deadlock prevention. CLEAN.

### File 10: `erisc.cc` (170 lines)
22. **Routing enable wait (line 106)**: Context-switches during wait. Host must enable. CLEAN.
23. **IRAM DMA wait (line 64)**: Hardware register poll, guaranteed. CLEAN.
24. **Kernel dispatch loop (lines 116-167)**: Standard go-message protocol with `notify_dispatch_core_done`. CLEAN.

---

## Conclusion

The audit found **no CRITICAL or HIGH severity bugs**. Two MEDIUM findings exist:

1. **WFEWS**: Unbounded credit wait in mux forward path — real but by-design (assumes downstream progress). Requires perf benchmarking to fix safely.
2. **MUX-NOWH**: Dead code in legacy mux — not currently reachable. Cleanup opportunity.

The previously applied fixes (FIX HS1, FIX HS2, FIX ER1, FIX AH) address the actual high-severity race conditions. The remaining codebase is well-structured with proper termination checks, bounded waits, and hardware-guaranteed drain patterns.

No MEDIUM+ fixes applied in this audit — both MEDIUM findings require careful performance validation that exceeds scope of a minimal audit fix.
