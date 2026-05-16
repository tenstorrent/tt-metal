<!--
SUMMARY: Complete census of all while/for/do-while loops in EDM/ERISC/ETH firmware — exit conditions, hang risk, and hang rescue proposals
KEYWORDS: edm, fabric, erisc, hang, loop, census, termination, watchdog, rescue, firmware
SOURCE: Comprehensive serial adversarial audit of tt-metal nsexton/0-racecondition-hunt
SCOPE: 14 firmware files covering all ERISC kernel and inline header code
USE WHEN: Investigating a firmware hang — use WAYPOINT codes to locate the stuck loop, then consult RESCUE INFO
-->

# Firmware Loop Census v1

**Branch**: `nsexton/0-racecondition-hunt`
**Date**: 2026-05-10
**Auditor**: BrAIn (adversarial audit — assumes comments may be wrong, worst-case behavior)

---

## Table of Contents

1. [fabric_erisc_router.cpp](#1-fabric_erisc_routercpp)
2. [fabric_router_relay_extension.cpp](#2-fabric_router_relay_extensioncpp)
3. [fabric_router_mux_extension.cpp](#3-fabric_router_mux_extensioncpp)
4. [fabric_router_udm_mux_extension.cpp](#4-fabric_router_udm_mux_extensioncpp)
5. [tt_fabric_mux.cpp](#5-tt_fabric_muxcpp)
6. [fabric_erisc_datamover_channels.hpp](#6-fabric_erisc_datamover_channelshpp)
7. [fabric_edm_packet_transmission.hpp](#7-fabric_edm_packet_transmissionhpp)
8. [fabric_router_flow_control.hpp](#8-fabric_router_flow_controlhpp)
9. [edm_handshake.hpp](#9-edm_handshakehpp)
10. [fabric_router_eth_handshake.hpp](#10-fabric_router_eth_handshakehpp)
11. [edm_fabric_worker_adapters.hpp](#11-edm_fabric_worker_adaptershpp)
12. [tt_fabric_utils.h](#12-tt_fabric_utilsh)
13. [fabric_erisc_router_speedy_path.hpp](#13-fabric_erisc_router_speedy_pathhpp)
14. [tt_fabric_mux.hpp](#14-tt_fabric_muxhpp)
15. [Summary: All WEAK/MISSING Loops](#15-summary-all-weakmissing-loops)
16. [Rescue Plan](#16-rescue-plan)
17. [Application-Level Rescue](#17-application-level-rescue)

---

## 1. fabric_erisc_router.cpp

### LOOP-001: TXQ drain after sender packet commit (line 622)

```
LOOP ID:    fabric_erisc_router.cpp:622 — run_sender_channel_step (non-speedy path)
CONDITION:  while (internal_::eth_txq_is_busy(sender_txq_id))
EXIT PATH:  ETH TX queue drains the committed packet
SETTER:     Hardware — ETH MAC DMA engine
BROKEN BY:  ETH link failure (link down, retrain in progress); ETH MAC hardware bug; stale TXQ state from prior ERISC soft-reset
WAYPOINT:   none
STRENGTH:   STRONG (hardware-guaranteed drain for committed packets)
HANG RISK:  LOW
RESCUE INFO: HW guarantees drain for committed packets. Only risk is ETH MAC failure. If link is being retrained this would be caught by the coordinated context switch mechanism. Post-reset stale TXQ is handled by init_handshake_info() flush guard.
FIX TAG:    LOOP-001
```

### LOOP-002: Coordinated context switch — master waits for INTENT_ACK (line 1442)

```
LOOP ID:    fabric_erisc_router.cpp:1442 — coordinated_context_switch_start_as_master
CONDITION:  while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() != INTENT_ACK)
EXIT PATH:  (a) erisc1 writes INTENT_ACK, or (b) termination signal
SETTER:     Local erisc1 (same chip, same core pair)
BROKEN BY:  erisc1 crashes/hangs before reading RETRAIN_INTENT; erisc1 is stuck in its own spin loop
WAYPOINT:   none
STRENGTH:   WEAK (termination signal provides escape on BH; on WH termination signal IS checked here)
HANG RISK:  MEDIUM — erisc1 must be alive and responsive
RESCUE INFO: Add a bounded iteration count with WAYPOINT("CSMA") on timeout. Host can detect via heartbeat stall on erisc0.
FIX TAG:    LOOP-002
```

### LOOP-003: Coordinated context switch — master waits for COMPLETE_ACK (line 1459)

```
LOOP ID:    fabric_erisc_router.cpp:1459 — coordinated_context_switch_finish_as_master
CONDITION:  while (read_stream_scratch_register<>() != COMPLETE_ACK)
EXIT PATH:  (a) erisc1 writes COMPLETE_ACK, or (b) termination signal
SETTER:     Local erisc1
BROKEN BY:  Same as LOOP-002 — erisc1 crashes after seeing RETRAIN_COMPLETE but before writing ACK
WAYPOINT:   none
STRENGTH:   WEAK (same as LOOP-002)
HANG RISK:  MEDIUM
RESCUE INFO: Same as LOOP-002 — add bounded iteration count with WAYPOINT("CSMF").
FIX TAG:    LOOP-003
```

### LOOP-004: Non-master waits for RETRAIN_COMPLETE (line 1486)

```
LOOP ID:    fabric_erisc_router.cpp:1486 — run_routing_without_noc_sync_coordinated_as_non_master
CONDITION:  while (read_stream_scratch_register<>() != RETRAIN_COMPLETE)
EXIT PATH:  (a) Master writes RETRAIN_COMPLETE, or (b) termination signal
SETTER:     Local erisc0 (master)
BROKEN BY:  Master crashes during retrain; retrain itself hangs (base firmware issue)
WAYPOINT:   none
STRENGTH:   WEAK (termination signal checked)
HANG RISK:  MEDIUM
RESCUE INFO: Add bounded iteration count with WAYPOINT("CSNS"). Master-side retrain is opaque from erisc1's perspective.
FIX TAG:    LOOP-004
```

### LOOP-005: Non-master waits for NORMAL_EXECUTION (line 1494)

```
LOOP ID:    fabric_erisc_router.cpp:1494 — run_routing_without_noc_sync_coordinated_as_non_master
CONDITION:  while (read_stream_scratch_register<>() != NORMAL_EXECUTION)
EXIT PATH:  (a) Master writes NORMAL_EXECUTION, or (b) termination signal
SETTER:     Local erisc0 (master)
BROKEN BY:  Master hangs after writing RETRAIN_COMPLETE but before writing NORMAL_EXECUTION (intervening crash)
WAYPOINT:   none
STRENGTH:   WEAK (termination signal checked)
HANG RISK:  MEDIUM
RESCUE INFO: Add bounded iteration count with WAYPOINT("CSNR").
FIX TAG:    LOOP-005
```

### LOOP-006: Retrain step — wait for host to clear RETRAIN command (line 2067)

```
LOOP ID:    fabric_erisc_router.cpp:2067 — run_retrain_step
CONDITION:  while (state_manager->command == RETRAIN && !got_immediate_termination_signal)
EXIT PATH:  (a) Host writes a different command (e.g. PAUSE), or (b) termination signal
SETTER:     Host (PCIe write to L1)
BROKEN BY:  Host crashes, host driver bug, host doesn't know retrain completed, PCIe link failure
WAYPOINT:   none
STRENGTH:   WEAK (termination signal provides escape, but host must send it)
HANG RISK:  MEDIUM — host-dependent
RESCUE INFO: Add WAYPOINT("RTRW") on timeout. Host watchdog can set PAUSE command or termination signal.
FIX TAG:    LOOP-006
```

### LOOP-007: Drain step — wait for host to clear DRAIN command (line 2092)

```
LOOP ID:    fabric_erisc_router.cpp:2092 — run_drain_step
CONDITION:  while (state_manager->command == DRAIN && !got_immediate_termination_signal)
EXIT PATH:  (a) Host writes different command, or (b) termination signal
SETTER:     Host
BROKEN BY:  Same as LOOP-006; also ASSERT(false) fires before this loop — drain is NOT IMPLEMENTED
WAYPOINT:   none
STRENGTH:   WEAK (but unreachable due to assert — dead code)
HANG RISK:  LOW (dead code; ASSERT(false) fires first)
RESCUE INFO: Mark as dead code. If assert is compiled out in release, this becomes a real WEAK spin.
FIX TAG:    LOOP-007
```

### LOOP-008: Pause command loop (line 2105)

```
LOOP ID:    fabric_erisc_router.cpp:2105 — execute_pause_command
CONDITION:  while (keep_running_pause && !got_immediate_termination_signal)
EXIT PATH:  (a) Host writes RUN command, or (b) termination signal
SETTER:     Host (writes RouterCommand::RUN to L1)
BROKEN BY:  Host never sends RUN; host crashes
WAYPOINT:   none
STRENGTH:   WEAK (termination signal provides escape; host must eventually send RUN or terminate)
HANG RISK:  LOW-MEDIUM — designed to wait for host; host has termination signal as escape hatch
RESCUE INFO: Add periodic WAYPOINT("PAUS") so host can see router is paused. Host watchdog should timeout if PAUSED too long.
FIX TAG:    LOOP-008
```

### LOOP-009: Main run loop — erisc0 state machine (line 2291)

```
LOOP ID:    fabric_erisc_router.cpp:2291 — execute_main_loop lambda (inside run_main_loop)
CONDITION:  while (continue_running_main_run_loop) — checks termination signal + non-run command pending
EXIT PATH:  (a) termination signal, or (b) state manager has non-run command
SETTER:     Host (termination signal or command change)
BROKEN BY:  Infinite spin if inner loop body has a hang (e.g. sender step TXQ drain)
WAYPOINT:   none (but FABRIC_KERNEL_HEARTBEAT_ADDR is incremented periodically)
STRENGTH:   STRONG (termination signal + command check on every outer iteration)
HANG RISK:  LOW — the loop itself exits cleanly; risk is from inner called functions
RESCUE INFO: Heartbeat monitoring. This loop is the expected steady-state.
FIX TAG:    LOOP-009
```

### LOOP-010: Inner iteration loop — between context switch checks (line 2330)

```
LOOP ID:    fabric_erisc_router.cpp:2330 — for (i < iterations_between_ctx_switch_and_teardown_checks)
CONDITION:  Bounded for loop, typically small constant
EXIT PATH:  Loop counter
SETTER:     Local — compile-time constant
BROKEN BY:  Only if called functions inside the loop hang (e.g. sender/receiver step functions)
WAYPOINT:   none
STRENGTH:   STRONG (bounded)
HANG RISK:  NONE (for the loop itself)
RESCUE INFO: N/A — bounded loop
FIX TAG:    LOOP-010
```

### LOOP-011: Erisc0 outer state machine loop (line 2673)

```
LOOP ID:    fabric_erisc_router.cpp:2673 — run_main_loop (MY_ERISC_ID == 0)
CONDITION:  while (!got_immediate_termination_signal)
EXIT PATH:  Termination signal from host
SETTER:     Host
BROKEN BY:  Host never sends termination; inner execute_main_loop hangs
WAYPOINT:   none (heartbeat covers this)
STRENGTH:   STRONG (termination signal checked per outer iteration)
HANG RISK:  LOW
RESCUE INFO: Heartbeat-based detection. This is the designed steady-state dispatch loop.
FIX TAG:    LOOP-011
```

### LOOP-012: Erisc1 outer run loop (line 2700)

```
LOOP ID:    fabric_erisc_router.cpp:2700 — run_main_loop (MY_ERISC_ID == 1)
CONDITION:  while (!got_immediate_termination_signal)
EXIT PATH:  Termination signal from host
SETTER:     Host
BROKEN BY:  Host never sends termination; inner execute_main_loop hangs
WAYPOINT:   none
STRENGTH:   STRONG (same as LOOP-011)
HANG RISK:  LOW
RESCUE INFO: Same as LOOP-011
FIX TAG:    LOOP-012
```

### LOOP-013: wait_for_static_connection_to_ready (line 2727)

```
LOOP ID:    fabric_erisc_router.cpp:2727 — wait_for_static_connection_to_ready (lambda)
CONDITION:  while (!connect_is_requested && [on BH: !got_immediate_termination_signal])
EXIT PATH:  Worker writes open_connection_value to semaphore; or termination signal (BH only)
SETTER:     Remote worker (via NOC write to L1 semaphore)
BROKEN BY:  Worker never connects (worker crashes, host doesn't launch worker, wrong address)
WAYPOINT:   WAYPOINT("SCRW") on watchdog timeout
STRENGTH:   WEAK on WH (no termination escape; watchdog reports but doesn't break out)
            WEAK on BH (termination signal provides escape, but worker must connect for progress)
HANG RISK:  HIGH on WH — no escape if worker never connects
RESCUE INFO: FIXED: FIX LT4 — watchdog counter fires WAYPOINT("SCRW"). Still no break-out on WH.
             Proposed: add a configurable max-iteration break-out that sets an error flag.
FIX TAG:    LOOP-013 (FIXED: FIX LT4 — watchdog added, but still no break-out on WH)
```

### LOOP-014: wait_for_other_local_erisc — master side (line 2879)

```
LOOP ID:    fabric_erisc_router.cpp:2879 — wait_for_other_local_erisc (IS_TEARDOWN_MASTER)
CONDITION:  while ((register & 0x1FFF) != multi_erisc_sync_step2_value)
EXIT PATH:  (a) Other ERISC writes sync value, or (b) watchdog_count >= kSyncMaxIter → break
SETTER:     Local erisc1
BROKEN BY:  Erisc1 crashed before reaching teardown sync
WAYPOINT:   WAYPOINT("TSYN") on timeout
STRENGTH:   STRONG (bounded — breaks out after ~100M iterations)
HANG RISK:  NONE (bounded break-out)
RESCUE INFO: FIXED: FIX LT2 — bounded break-out with WAYPOINT("TSYN")
FIX TAG:    LOOP-014 (FIXED: FIX LT2)
```

### LOOP-015: wait_for_other_local_erisc — non-master side (line 2890)

```
LOOP ID:    fabric_erisc_router.cpp:2890 — wait_for_other_local_erisc (!IS_TEARDOWN_MASTER)
CONDITION:  while ((register & 0x1FFF) != multi_erisc_sync_start_value)
EXIT PATH:  (a) Master writes sync value, or (b) watchdog_count >= kSyncMaxIter → break
SETTER:     Local erisc0
BROKEN BY:  Erisc0 crashed before reaching teardown sync
WAYPOINT:   WAYPOINT("TSYN") on timeout
STRENGTH:   STRONG (bounded break-out)
HANG RISK:  NONE
RESCUE INFO: FIXED: FIX LT2
FIX TAG:    LOOP-015 (FIXED: FIX LT2)
```

### LOOP-016 through LOOP-030: Bounded for loops over channel arrays

Lines 438, 481, 1413, 2859, 2985, 2992, 3156, 3161, 3172, 3186, 3189, 3192, 3204, 3214, 3217, 3220, 3231, 3321.

```
LOOP IDs:   fabric_erisc_router.cpp: various — for loops over NUM_SENDER_CHANNELS, NUM_RECEIVER_CHANNELS, etc.
CONDITION:  for (i = 0; i < N; i++) where N is a compile-time constant (typically 1-4)
EXIT PATH:  Loop counter
STRENGTH:   STRONG (bounded by compile-time constant)
HANG RISK:  NONE
FIX TAG:    LOOP-016..LOOP-030 (SAFE — bounded for loops)
```

### LOOP-031 through LOOP-035: Downstream EDM bit-scan loops

Lines 3351, 3437, 3680, 3735, 3756.

```
LOOP IDs:   fabric_erisc_router.cpp: various — while (has_downstream_edm)
CONDITION:  while (has_downstream_edm) where has_downstream_edm >>= 1 each iteration
EXIT PATH:  Mask shifts to 0 (at most 4 iterations for 4-bit mask)
STRENGTH:   STRONG (bounded — mask is ANDed with 0xF, max 4 bits)
HANG RISK:  NONE
FIX TAG:    LOOP-031..LOOP-035 (SAFE — bounded bit-scan)
```

---

## 2. fabric_router_relay_extension.cpp

### LOOP-036: wait_for_static_connection_to_ready (line 150)

```
LOOP ID:    fabric_router_relay_extension.cpp:150 — wait_for_static_connection_to_ready
CONDITION:  while (!connect_is_requested && [BH: !got_termination_signal])
EXIT PATH:  Worker writes open_connection_value; or termination signal (BH only)
SETTER:     Remote worker (via NOC write)
BROKEN BY:  Worker never connects
WAYPOINT:   WAYPOINT("SCRW") on watchdog timeout
STRENGTH:   WEAK on WH (watchdog reports but doesn't break out)
HANG RISK:  HIGH on WH
RESCUE INFO: FIXED: FIX LT4 — watchdog counter fires WAYPOINT("SCRW"). No break-out on WH.
FIX TAG:    LOOP-036 (FIXED: FIX LT4)
```

### LOOP-037: wait_for_mux_endpoint_ready (line 176)

```
LOOP ID:    fabric_router_relay_extension.cpp:176 — wait_for_mux_endpoint_ready
CONDITION:  for (i = 0; i < max_poll_iterations; ++i) where max_poll_iterations=1'000'000
EXIT PATH:  (a) Mux status becomes READY_FOR_TRAFFIC, (b) loop counter exhausted, (c) termination signal (BH only)
SETTER:     Remote mux kernel (writes READY_FOR_TRAFFIC to L1 status)
BROKEN BY:  Mux kernel hasn't started yet or crashed during init
WAYPOINT:   none
STRENGTH:   STRONG (bounded by 1M iterations — falls through on timeout)
HANG RISK:  NONE (bounded)
RESCUE INFO: Bounded. Caller handles fall-through (host Phase 4 recovery).
FIX TAG:    LOOP-037 (SAFE — bounded)
```

### LOOP-038: Local mux status poll (line 568)

```
LOOP ID:    fabric_router_relay_extension.cpp:568 — kernel_main (local mux status wait)
CONDITION:  for (i = 0; i < 1'000'000U; ++i)
EXIT PATH:  (a) Mux status == READY_FOR_TRAFFIC, (b) loop counter exhausted, (c) termination signal (BH only)
SETTER:     Local mux kernel (same core, writes to L1 status)
BROKEN BY:  Mux init hangs
WAYPOINT:   none
STRENGTH:   STRONG (bounded by 1M iterations)
HANG RISK:  NONE (bounded)
FIX TAG:    LOOP-038 (SAFE — bounded)
```

### LOOP-039: Main relay processing loop (line 607)

```
LOOP ID:    fabric_router_relay_extension.cpp:607 — kernel_main main loop
CONDITION:  while (!got_immediate_termination_signal<true>)
EXIT PATH:  Termination signal from host (via mux kernel → relay termination address)
SETTER:     Host (indirectly via mux kernel)
BROKEN BY:  Mux kernel never relays termination signal
WAYPOINT:   none
STRENGTH:   STRONG (termination signal checked)
HANG RISK:  LOW — risk is from inner processing functions, not the loop itself
RESCUE INFO: Inner for loop (line 608) is bounded. Main risk is from process_inbound_packet internals.
FIX TAG:    LOOP-039
```

### LOOP-040 through LOOP-043: Bounded for loops

Lines 329, 508, 521, 581, 593.

```
STRENGTH:   STRONG (bounded by compile-time constants or small constants)
HANG RISK:  NONE
FIX TAG:    LOOP-040..LOOP-043 (SAFE)
```

---

## 3. fabric_router_mux_extension.cpp

### LOOP-044: wait_for_static_connection_to_ready (line 124)

```
LOOP ID:    fabric_router_mux_extension.cpp:124 — wait_for_static_connection_to_ready
CONDITION:  while (!connect_is_requested && [BH: !got_termination_signal])
EXIT PATH:  Worker writes open_connection_value; or termination signal (BH only)
SETTER:     Remote worker (via NOC write)
WAYPOINT:   WAYPOINT("SCRW") on watchdog timeout
STRENGTH:   WEAK on WH
HANG RISK:  HIGH on WH
RESCUE INFO: FIXED: FIX LT4 — watchdog counter fires WAYPOINT("SCRW"). No break-out on WH.
FIX TAG:    LOOP-044 (FIXED: FIX LT4)
```

### LOOP-045: Main mux processing loop (line 374)

```
LOOP ID:    fabric_router_mux_extension.cpp:374 — kernel_main main loop
CONDITION:  while (!got_immediate_termination_signal<true>)
EXIT PATH:  Termination signal from host
SETTER:     Host
WAYPOINT:   RISC_POST_HEARTBEAT on IDLE_ERISC builds
STRENGTH:   STRONG (termination signal checked)
HANG RISK:  LOW
RESCUE INFO: Heartbeat monitoring. Inner loops are bounded.
FIX TAG:    LOOP-045
```

### LOOP-046 through LOOP-051: Bounded for loops

Lines 227, 230, 240, 293, 314, 334, 357, 363, 375, 377, 389.

```
STRENGTH:   STRONG (all bounded by compile-time constants)
HANG RISK:  NONE
FIX TAG:    LOOP-046..LOOP-051 (SAFE)
```

---

## 4. fabric_router_udm_mux_extension.cpp

### LOOP-052: wait_for_static_connection_to_ready (line 186)

```
LOOP ID:    fabric_router_udm_mux_extension.cpp:186 — wait_for_static_connection_to_ready
CONDITION:  while (!connect_is_requested && [BH: !got_termination_signal])
EXIT PATH:  Worker writes open_connection_value; or termination signal (BH only)
SETTER:     Remote worker
WAYPOINT:   WAYPOINT("SCRW") on watchdog timeout
STRENGTH:   WEAK on WH
HANG RISK:  HIGH on WH
RESCUE INFO: FIXED: FIX LT4
FIX TAG:    LOOP-052 (FIXED: FIX LT4)
```

### LOOP-053: wait_for_mux_endpoint_ready (line 208)

```
LOOP ID:    fabric_router_udm_mux_extension.cpp:208 — wait_for_mux_endpoint_ready
CONDITION:  for (i = 0; i < max_poll_iterations; ++i) where max=1'000'000
EXIT PATH:  (a) Status == READY_FOR_TRAFFIC, (b) loop counter exhausted
SETTER:     Remote mux kernel
WAYPOINT:   none
STRENGTH:   STRONG (bounded by 1M iterations)
HANG RISK:  NONE
RESCUE INFO: Falls through on timeout.
FIX TAG:    LOOP-053 (SAFE — bounded)
```

### LOOP-054: Main UDM mux processing loop (line 555)

```
LOOP ID:    fabric_router_udm_mux_extension.cpp:555 — kernel_main main loop
CONDITION:  while (!got_immediate_termination_signal<true>)
EXIT PATH:  Termination signal from host
SETTER:     Host
WAYPOINT:   none
STRENGTH:   STRONG (termination signal checked)
HANG RISK:  LOW
RESCUE INFO: Inner loops are bounded.
FIX TAG:    LOOP-054
```

### LOOP-055 through LOOP-063: Bounded for loops

Lines 316, 407, 428, 449, 471, 519, 530, 537, 543, 549, 556, 558, 572, 586.

```
STRENGTH:   STRONG (all bounded)
HANG RISK:  NONE
FIX TAG:    LOOP-055..LOOP-063 (SAFE)
```

---

## 5. tt_fabric_mux.cpp

### LOOP-064: wait_for_static_connection_to_ready — DEAD CODE (line 60)

```
LOOP ID:    tt_fabric_mux.cpp:60 — wait_for_static_connection_to_ready (DEAD CODE)
CONDITION:  while (!connect_is_requested(*worker_interface.connection_live_semaphore))
EXIT PATH:  Worker writes open_connection_value
SETTER:     Remote worker
BROKEN BY:  Worker never connects — no termination signal check, no watchdog, no escape
WAYPOINT:   none
STRENGTH:   MISSING (no timeout, no termination check, no watchdog)
HANG RISK:  HIGH (but DEAD CODE — never called from kernel_main)
RESCUE INFO: FIXED: FIX LT3 — marked as dead code. Should be deleted.
FIX TAG:    LOOP-064 (FIXED: FIX LT3 — dead code)
```

### LOOP-065: Main mux processing loop (line 247)

```
LOOP ID:    tt_fabric_mux.cpp:247 — kernel_main main loop
CONDITION:  while (!got_immediate_termination_signal<true>)
EXIT PATH:  (a) Immediate termination signal, or (b) graceful termination + all channels drained
SETTER:     Host (termination signal)
WAYPOINT:   RISC_POST_HEARTBEAT on IDLE_ERISC builds
STRENGTH:   STRONG (both immediate and graceful termination checked)
HANG RISK:  LOW
RESCUE INFO: Graceful drain logic could stall if a channel never drains (sender stopped sending, credits lost). In that case, immediate termination signal is the rescue. Host watchdog should escalate graceful to immediate.
FIX TAG:    LOOP-065
```

### LOOP-066 through LOOP-071: Bounded for loops

Lines 161, 195, 209, 251, 254, 264, 265, 266, 277.

```
STRENGTH:   STRONG (all bounded)
HANG RISK:  NONE
FIX TAG:    LOOP-066..LOOP-071 (SAFE)
```

---

## 6. fabric_erisc_datamover_channels.hpp

### LOOP-072 through LOOP-075: Bounded buffer initialization loops

Lines 60, 64, 167, 171.

```
LOOP IDs:   fabric_erisc_datamover_channels.hpp — for loops over NUM_BUFFERS and header word counts
CONDITION:  for (i = 0; i < N; i++) where N is small compile-time constant
STRENGTH:   STRONG (bounded)
HANG RISK:  NONE
FIX TAG:    LOOP-072..LOOP-075 (SAFE)
```

---

## 7. fabric_edm_packet_transmission.hpp

### LOOP-076: NOC TRID flush — downstream noc (line 122-128)

```
LOOP ID:    fabric_edm_packet_transmission.hpp:122-128 — flush_write_to_noc_pipeline
CONDITION:  while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid))
EXIT PATH:  NOC transaction with given TRID completes
SETTER:     Hardware — NOC DMA engine
BROKEN BY:  (a) NOC hang (target tile unresponsive), (b) TRID never issued (software bug — flushing a TRID that was never used), (c) NOC arbitration deadlock
WAYPOINT:   none
STRENGTH:   WEAK (hardware-dependent; NOC hangs are possible under certain conditions)
HANG RISK:  MEDIUM — NOC deadlocks are rare but have been observed; TRID bugs can cause this
RESCUE INFO: These loops run inside a for(trid) loop so multiple TRIDs are flushed. If one TRID hangs, all subsequent ones also hang. Add a bounded iteration count per-TRID with WAYPOINT("NFLT") (NOC Flush TRID). Host can detect via heartbeat stall.
FIX TAG:    LOOP-076
```

### LOOP-077: NOC TRID flush — local chip noc (line 125/128, 134-140)

```
LOOP ID:    fabric_edm_packet_transmission.hpp:125+128,134-140 — flush_write_to_noc_pipeline (non-deadlock-avoidance path)
CONDITION:  Same as LOOP-076 but for edm_to_local_chip_noc
EXIT PATH:  Same as LOOP-076
SETTER:     Hardware — NOC DMA engine
BROKEN BY:  Same as LOOP-076
WAYPOINT:   none
STRENGTH:   WEAK
HANG RISK:  MEDIUM
RESCUE INFO: Same as LOOP-076
FIX TAG:    LOOP-077
```

### LOOP-078, LOOP-079: Bounded for loops

Lines 411, 419 — scatter write chunk iteration.

```
STRENGTH:   STRONG (bounded by EXT, compile-time constant)
HANG RISK:  NONE
FIX TAG:    LOOP-078, LOOP-079 (SAFE)
```

---

## 8. fabric_router_flow_control.hpp

### LOOP-080: receiver_send_completion_ack TXQ wait (line 247)

```
LOOP ID:    fabric_router_flow_control.hpp:247 — receiver_send_completion_ack<CHECK_BUSY=true>
CONDITION:  while (internal_::eth_txq_is_busy(receiver_txq_id))
EXIT PATH:  ETH TX queue drains
SETTER:     Hardware — ETH MAC
BROKEN BY:  ETH link failure; stale TXQ state
WAYPOINT:   none
STRENGTH:   STRONG (hardware-guaranteed for committed packets)
HANG RISK:  LOW — same as LOOP-001
RESCUE INFO: Same analysis as TXQ drains in LOOP-001.
FIX TAG:    LOOP-080
```

### LOOP-081: receiver_send_received_ack TXQ wait (line 260)

```
LOOP ID:    fabric_router_flow_control.hpp:260 — receiver_send_received_ack<CHECK_BUSY=true>
CONDITION:  while (internal_::eth_txq_is_busy(receiver_txq_id))
EXIT PATH:  Same as LOOP-080
SETTER:     Hardware
STRENGTH:   STRONG
HANG RISK:  LOW
FIX TAG:    LOOP-081
```

### LOOP-082 through LOOP-087: Bounded for loops

Lines 21, 58, 92, 105, 194, 209.

```
STRENGTH:   STRONG (bounded)
HANG RISK:  NONE
FIX TAG:    LOOP-082..LOOP-087 (SAFE)
```

---

## 9. edm_handshake.hpp

### LOOP-088: init_handshake_info — stale TXQ flush wait (line ~82)

```
LOOP ID:    edm_handshake.hpp:~82 — init_handshake_info (conditional flush)
CONDITION:  while (eth_txq_is_busy()) { } — ONLY entered if eth_txq_is_busy() was true (guarded)
EXIT PATH:  ETH_TXQ_CMD_FLUSH completes
SETTER:     Hardware — ETH MAC processes flush command
BROKEN BY:  Flush command itself hangs (HW bug)
WAYPOINT:   none
STRENGTH:   STRONG (guarded entry; flush is hardware-guaranteed; stale-TXQ-on-idle bug is avoided by the guard)
HANG RISK:  LOW
RESCUE INFO: The guard prevents the nop-flush-hangs-forever scenario. Only risk is genuine HW failure.
FIX TAG:    LOOP-088
```

### LOOP-089: sender_side_handshake (line ~113)

```
LOOP ID:    edm_handshake.hpp:~113 — sender_side_handshake
CONDITION:  while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE)
EXIT PATH:  Remote ERISC writes MAGIC_HANDSHAKE_VALUE via eth_send_packet
SETTER:     Remote ERISC (peer on ethernet link)
BROKEN BY:  Remote ERISC not started; remote ERISC crashed; ETH link down; simultaneous-sender race (FIXED by post-loop send)
WAYPOINT:   none
STRENGTH:   WEAK (no timeout, no termination signal check; relies entirely on remote ERISC being alive)
HANG RISK:  HIGH — if remote ERISC never starts or has link issues, this spins forever
RESCUE INFO: FIXED: FIX HS1 — post-exit final send handles simultaneous-sender race. But the loop itself is still unbounded. Context switch via run_routing() is called periodically. Host watchdog must detect via heartbeat stall.
FIX TAG:    LOOP-089 (FIXED: FIX HS1 — race fix; loop still unbounded)
```

### LOOP-090: receiver_side_handshake (line ~141)

```
LOOP ID:    edm_handshake.hpp:~141 — receiver_side_handshake
CONDITION:  while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE)
EXIT PATH:  Remote ERISC writes MAGIC_HANDSHAKE_VALUE
SETTER:     Remote ERISC (peer)
BROKEN BY:  Same as LOOP-089
WAYPOINT:   none
STRENGTH:   WEAK (no timeout, no termination check)
HANG RISK:  HIGH — same as LOOP-089
RESCUE INFO: Context switch via run_routing() called periodically. No watchdog. Host must detect via heartbeat.
FIX TAG:    LOOP-090
```

### LOOP-091: deprecated::sender_side_start — eth_txq_is_busy (line ~200)

```
LOOP ID:    edm_handshake.hpp:~200 — deprecated::sender_side_start
CONDITION:  while (eth_txq_is_busy())
EXIT PATH:  ETH TX queue drains
SETTER:     Hardware
STRENGTH:   STRONG (hardware-guaranteed)
HANG RISK:  LOW
RESCUE INFO: This is in deprecated code path.
FIX TAG:    LOOP-091
```

### LOOP-092: deprecated::receiver_side_finish — eth_txq_is_busy (line ~240)

```
LOOP ID:    edm_handshake.hpp:~240 — deprecated::receiver_side_finish
CONDITION:  while (eth_txq_is_busy())
EXIT PATH:  ETH TX queue drains
SETTER:     Hardware
STRENGTH:   STRONG
HANG RISK:  LOW
RESCUE INFO: Deprecated code path.
FIX TAG:    LOOP-092
```

**Note**: `eth_wait_receiver_done` (called from deprecated::sender_side_start, line 202) and `eth_wait_for_bytes` (called from deprecated::receiver_side_finish, line 237) are defined in `dataflow_api.h` and contain their own while loops waiting for `erisc_info->channels[0].bytes_sent`. These are WEAK loops (wait for remote ERISC to send/ack) but are in the deprecated code path and bounded by context switch timeout.

### LOOP-093: eth_wait_receiver_done (dataflow_api.h:289)

```
LOOP ID:    dataflow_api.h:289 — eth_wait_receiver_done (called from deprecated handshake)
CONDITION:  while (erisc_info->channels[0].bytes_sent != 0)
EXIT PATH:  Remote ERISC acknowledges receipt (clears bytes_sent)
SETTER:     Remote ERISC
BROKEN BY:  Remote ERISC not running, link down
WAYPOINT:   none
STRENGTH:   WEAK (no timeout, no termination check; context switch via run_routing)
HANG RISK:  MEDIUM (deprecated path — should be rare)
RESCUE INFO: In deprecated code. Context switch keeps base firmware alive.
FIX TAG:    LOOP-093
```

### LOOP-094: eth_wait_for_bytes (dataflow_api.h:414)

```
LOOP ID:    dataflow_api.h:414 — eth_wait_for_bytes (called from deprecated handshake)
CONDITION:  while (erisc_info->channels[0].bytes_sent != num_bytes)
EXIT PATH:  Remote ERISC sends expected bytes
SETTER:     Remote ERISC
STRENGTH:   WEAK (same as LOOP-093)
HANG RISK:  MEDIUM (deprecated path)
FIX TAG:    LOOP-094
```

---

## 10. fabric_router_eth_handshake.hpp

### LOOP-095: fabric_sender_side_handshake (line ~42)

```
LOOP ID:    fabric_router_eth_handshake.hpp:42 — fabric_sender_side_handshake
CONDITION:  while (local_value != MAGIC && [BH: !got_termination_signal])
EXIT PATH:  (a) Remote ERISC writes MAGIC, (b) termination signal (BH only)
SETTER:     Remote ERISC (peer on ETH link)
BROKEN BY:  Remote ERISC not started; link down
WAYPOINT:   WAYPOINT("HSST") on watchdog timeout
STRENGTH:   WEAK on WH (watchdog reports but no break-out; no termination check)
            WEAK on BH (termination provides escape)
HANG RISK:  HIGH on WH — no escape if peer never responds
RESCUE INFO: FIXED: FIX HS2 + FIX LT4 — post-loop final send + watchdog WAYPOINT("HSST"). Still no break-out on WH.
FIX TAG:    LOOP-095 (FIXED: FIX HS2 + FIX LT4)
```

### LOOP-096: fabric_receiver_side_handshake (line ~93)

```
LOOP ID:    fabric_router_eth_handshake.hpp:93 — fabric_receiver_side_handshake
CONDITION:  while (local_value != MAGIC && [BH: !got_termination_signal])
EXIT PATH:  (a) Remote ERISC writes MAGIC, (b) termination signal (BH only)
SETTER:     Remote ERISC
WAYPOINT:   WAYPOINT("HSRT") on watchdog timeout
STRENGTH:   WEAK on WH (watchdog reports but no break-out)
HANG RISK:  HIGH on WH
RESCUE INFO: FIXED: FIX LT4 — watchdog WAYPOINT("HSRT"). No break-out on WH.
FIX TAG:    LOOP-096 (FIXED: FIX LT4)
```

---

## 11. edm_fabric_worker_adapters.hpp

### LOOP-097: wait_for_empty_write_slot (line 314)

```
LOOP ID:    edm_fabric_worker_adapters.hpp:314 — wait_for_empty_write_slot
CONDITION:  while (!this->edm_has_space_for_packet<1>())
EXIT PATH:  Downstream EDM frees a buffer slot (credits arrive)
SETTER:     Downstream EDM (via stream register or NOC write)
BROKEN BY:  Downstream EDM dead, credits lost, flow control bug
WAYPOINT:   WAYPOINT("FWSW") on entry, WAYPOINT("FWST") on watchdog timeout, WAYPOINT("FWSD") on exit
STRENGTH:   WEAK (watchdog reports but doesn't break out — caller cannot proceed without free slot)
HANG RISK:  HIGH — if downstream EDM is dead, this spins forever
RESCUE INFO: FIXED: FIX LT1 — watchdog counter added. Still no break-out (by design: caller needs the slot). Host must detect via WAYPOINT("FWST") and force-terminate.
FIX TAG:    LOOP-097 (FIXED: FIX LT1)
```

### LOOP-098: close_finish — wait for EDM ACK (line 542)

```
LOOP ID:    edm_fabric_worker_adapters.hpp:542 — close_finish
CONDITION:  for (i = 0; i < kCloseFinishMaxIter; ++i) where kCloseFinishMaxIter=5'000'000
EXIT PATH:  (a) EDM writes 1 to worker_teardown_addr, or (b) loop counter exhausted
SETTER:     ETH router (via NOC write to L1)
BROKEN BY:  ETH router never ACKs (ARC blocked the NOC write, router crashed)
WAYPOINT:   WAYPOINT("FCFW") on entry, WAYPOINT("FCFD") on exit
STRENGTH:   STRONG (bounded by 5M iterations)
HANG RISK:  NONE (bounded)
RESCUE INFO: Falls through on timeout. Caller exits cleanly.
FIX TAG:    LOOP-098 (SAFE — bounded)
```

### LOOP-099: Buffer slot address initialization (line 240)

```
LOOP ID:    edm_fabric_worker_adapters.hpp:240 — constructor
CONDITION:  for (i = 0; i < EDM_NUM_BUFFER_SLOTS; ++i)
STRENGTH:   STRONG (bounded)
HANG RISK:  NONE
FIX TAG:    LOOP-099 (SAFE)
```

---

## 12. tt_fabric_utils.h

### LOOP-100: wait_for_notification (line ~88)

```
LOOP ID:    tt_fabric_utils.h:~88 — wait_for_notification
CONDITION:  while (*poll_addr != value && [BH: !got_termination_signal])
EXIT PATH:  (a) Target value arrives at poll address, (b) termination signal (BH only)
SETTER:     Remote entity (depends on caller — could be another router, host, or worker)
BROKEN BY:  Notifier never sends the expected value
WAYPOINT:   WAYPOINT("WNTO") on watchdog timeout
STRENGTH:   WEAK on WH (watchdog reports but no break-out; no termination check)
HANG RISK:  HIGH on WH
RESCUE INFO: FIXED: FIX LT4 — watchdog WAYPOINT("WNTO"). Still no break-out on WH. Context switch via run_routing() keeps base firmware alive.
FIX TAG:    LOOP-100 (FIXED: FIX LT4)
```

---

## 13. fabric_erisc_router_speedy_path.hpp

### LOOP-101: TXQ drain in speedy sender path (line 141)

```
LOOP ID:    fabric_erisc_router_speedy_path.hpp:141 — run_sender_channel_step_speedy
CONDITION:  while (busy) where busy = internal_::eth_txq_is_busy(sender_txq_id)
EXIT PATH:  ETH TX queue drains
SETTER:     Hardware — ETH MAC
BROKEN BY:  ETH link failure (same as LOOP-001)
WAYPOINT:   none
STRENGTH:   STRONG (hardware-guaranteed drain for committed packets)
HANG RISK:  LOW
RESCUE INFO: Same as LOOP-001.
FIX TAG:    LOOP-101
```

---

## 14. tt_fabric_mux.hpp

No loops found in this file (39 lines, header-only declarations).

---

## 15. Summary: All WEAK/MISSING Loops

```
FIX TAG    FILE                                  LINE   CONDITION SUMMARY                        WH RISK  BH RISK  STATUS
---------- ------------------------------------- ------ ----------------------------------------- -------- -------- ---------
LOOP-002   fabric_erisc_router.cpp               1442   ctx switch master→INTENT_ACK              MEDIUM   MEDIUM   NEW
LOOP-003   fabric_erisc_router.cpp               1459   ctx switch master→COMPLETE_ACK             MEDIUM   MEDIUM   NEW
LOOP-004   fabric_erisc_router.cpp               1486   ctx switch non-master→RETRAIN_COMPLETE     MEDIUM   MEDIUM   NEW
LOOP-005   fabric_erisc_router.cpp               1494   ctx switch non-master→NORMAL_EXECUTION     MEDIUM   MEDIUM   NEW
LOOP-006   fabric_erisc_router.cpp               2067   retrain wait for host cmd change           MEDIUM   MEDIUM   NEW
LOOP-008   fabric_erisc_router.cpp               2105   pause wait for host RUN                    LOW-MED  LOW-MED  NEW
LOOP-013   fabric_erisc_router.cpp               2727   wait_for_static_connection (router)        HIGH     MEDIUM   FIX LT4
LOOP-036   relay_extension.cpp                   150    wait_for_static_connection (relay)         HIGH     MEDIUM   FIX LT4
LOOP-044   mux_extension.cpp                     124    wait_for_static_connection (mux)           HIGH     MEDIUM   FIX LT4
LOOP-052   udm_mux_extension.cpp                 186    wait_for_static_connection (udm_mux)       HIGH     MEDIUM   FIX LT4
LOOP-064   tt_fabric_mux.cpp                     60     wait_for_static_connection (DEAD CODE)     HIGH     HIGH     FIX LT3
LOOP-076   fabric_edm_packet_transmission.hpp    122    NOC TRID flush (downstream)                MEDIUM   MEDIUM   NEW
LOOP-077   fabric_edm_packet_transmission.hpp    134    NOC TRID flush (local chip)                MEDIUM   MEDIUM   NEW
LOOP-089   edm_handshake.hpp                     ~113   sender_side_handshake                      HIGH     HIGH     FIX HS1
LOOP-090   edm_handshake.hpp                     ~141   receiver_side_handshake                    HIGH     HIGH     NEW*
LOOP-093   dataflow_api.h                        289    eth_wait_receiver_done (deprecated)         MEDIUM   MEDIUM   deprecated
LOOP-094   dataflow_api.h                        414    eth_wait_for_bytes (deprecated)             MEDIUM   MEDIUM   deprecated
LOOP-095   fabric_router_eth_handshake.hpp       42     fabric_sender_side_handshake               HIGH     MEDIUM   FIX HS2+LT4
LOOP-096   fabric_router_eth_handshake.hpp       93     fabric_receiver_side_handshake              HIGH     MEDIUM   FIX LT4
LOOP-097   edm_fabric_worker_adapters.hpp        314    wait_for_empty_write_slot                   HIGH     HIGH     FIX LT1
LOOP-100   tt_fabric_utils.h                     ~88    wait_for_notification                       HIGH     MEDIUM   FIX LT4
```

**Notes:**
- `*` LOOP-090: receiver_side_handshake has no FIX applied (no post-loop send needed for receiver, no watchdog added)
- "FIX LT4" loops have watchdog WAYPOINTs but still no break-out on WH
- "NEW" loops have neither watchdog nor break-out

---

## 16. Rescue Plan

### Priority 1: HIGH HANG RISK — New loops needing fixes

#### P1-A: NOC TRID flush loops (LOOP-076, LOOP-077)

**Problem**: `flush_write_to_noc_pipeline()` contains 2-6 tight `while` loops that spin until each TRID is flushed. If any NOC transaction hangs (target tile unresponsive, NOC arbitration deadlock, TRID never issued), all subsequent flushes in the same for-loop also hang.

**Proposed fix**:
- Add per-TRID bounded iteration count (e.g. 1B iterations ~ 4-8s)
- Fire WAYPOINT("NFLT") on timeout with TRID index encoded in a scratch register
- Do NOT break out — the caller (receiver channel teardown) needs the flush to complete for correctness. But the WAYPOINT allows host-side diagnosis.

**Impact**: Receiver teardown path only. Affects all configurations.

#### P1-B: Coordinated context switch loops (LOOP-002 through LOOP-005)

**Problem**: Four loops in the retrain coordination protocol between erisc0 and erisc1. Each waits for the other ERISC to write a specific state value. If one ERISC crashes mid-protocol, the other spins forever. Termination signal IS checked on both WH and BH, but if the host never sends it (because it doesn't know a hang happened), the spin continues.

**Proposed fix**:
- Add bounded iteration count with WAYPOINT (e.g. "CSMA", "CSMF", "CSNS", "CSNR")
- After timeout, break out and set an error flag in the router state
- Host watchdog can detect via heartbeat stall + WAYPOINT code

**Impact**: Only active when NUM_ACTIVE_ERISCS > 1 and retrain/context-switch is used.

#### P1-C: Retrain/pause host-wait loops (LOOP-006, LOOP-008)

**Problem**: Wait for host to change the router command. Host-dependent — if host crashes or hangs, these loops spin (but termination signal is checked on both archs).

**Proposed fix**:
- Add periodic WAYPOINT("RTRW" / "PAUS") so host can see the router's state
- No bounded break-out needed — termination signal is the designed escape hatch
- Host watchdog should escalate after timeout

**Impact**: Low priority — termination signal provides escape.

### Priority 2: Existing fixes that need strengthening

#### P2-A: All FIX LT4 loops on WH (LOOP-013, 036, 044, 052, 095, 096, 100)

**Problem**: These loops have watchdog WAYPOINTs added but still no actual break-out on WH (termination signal check is `#ifndef ARCH_WORMHOLE`). On WH, if the expected event never arrives, the loop spins forever (but fires WAYPOINT periodically).

**Proposed fix**: For each, evaluate whether a bounded break-out is safe:
- `wait_for_static_connection_to_ready`: Could break out and set channel to "not established" — but caller may not handle this gracefully.
- Handshake loops: Breaking out mid-handshake leaves the link in an indeterminate state. May be better to let host-side watchdog detect and force-reset.
- `wait_for_notification`: Used during init coordination. Breaking out could cause init ordering violations.

**Recommendation**: For WH, rely on host-side watchdog to detect WAYPOINT codes and force-terminate. Adding break-outs requires careful analysis of all callers.

#### P2-B: sender_side_handshake / receiver_side_handshake base versions (LOOP-089, LOOP-090)

**Problem**: Base versions in `edm_handshake.hpp` have NO watchdog, NO termination check. They rely entirely on the remote ERISC being alive. FIX HS1 added the post-loop send but the loop itself is still unbounded.

**Note**: These base versions are used by non-fabric EDM kernels. The fabric-specific versions in `fabric_router_eth_handshake.hpp` (LOOP-095, LOOP-096) DO have watchdogs. If all fabric code paths use the `fabric_*` variants, then LOOP-089/090 are only reached by legacy/deprecated callers.

**Proposed fix**: Add watchdog WAYPOINTs to the base versions too, for defense-in-depth.

### Priority 3: Low-risk items for completeness

#### P3-A: TXQ drain loops (LOOP-001, LOOP-080, LOOP-081, LOOP-088, LOOP-091, LOOP-092, LOOP-101)

These are all hardware-guaranteed drains. Risk is extremely low (ETH MAC failure). No action needed.

#### P3-B: Deprecated handshake loops (LOOP-093, LOOP-094)

In deprecated code path. Risk is low. Consider adding a note that these are deprecated and should be removed with the legacy CCL ops cleanup.

---

## 17. Application-Level Rescue

### Host-Side Watchdog Detection

The following WAYPOINT codes are already in place for diagnosing hung loops:

```
WAYPOINT     Meaning                                    Source Loop
------------ ------------------------------------------ ----------
"SCRW"       Static-Connection-Ready Wait timeout        LOOP-013, 036, 044, 052
"HSST"       HandShake Sender Timeout                    LOOP-095
"HSRT"       HandShake Receiver Timeout                  LOOP-096
"WNTO"       Wait-for-Notification TimeOut               LOOP-100
"FWSW"       Free Write Slot Wait (entry)                LOOP-097
"FWST"       Free Write Slot Timeout                     LOOP-097
"FWSD"       Free Write Slot Done (exit)                 LOOP-097
"TSYN"       Teardown SYNc timeout                       LOOP-014, 015
"FCFW"       close_finish Wait (entry)                   LOOP-098
"FCFD"       close_finish Done (exit)                    LOOP-098
```

### Proposed new WAYPOINT codes:

```
WAYPOINT     Meaning                                    Source Loop
------------ ------------------------------------------ ----------
"CSMA"       Context Switch Master Ack timeout           LOOP-002
"CSMF"       Context Switch Master Finish timeout        LOOP-003
"CSNS"       Context Switch Non-master Start timeout     LOOP-004
"CSNR"       Context Switch Non-master Resume timeout    LOOP-005
"RTRW"       ReTRain Wait for host                       LOOP-006
"PAUS"       PAUSe waiting for host RUN                  LOOP-008
"NFLT"       NOC FLush TRID timeout                      LOOP-076, 077
"HSSB"       HandShake Sender Base timeout               LOOP-089
"HSRB"       HandShake Receiver Base timeout             LOOP-090
```

### Host Watchdog Protocol

1. **Detection**: Poll `FABRIC_KERNEL_HEARTBEAT_ADDR` for each ERISC. If heartbeat stalls for >5s, the ERISC is hung.
2. **Diagnosis**: Read the WAYPOINT register to determine which loop is stuck. Cross-reference with this census.
3. **Recovery**:
   - Set `TerminationSignal::IMMEDIATELY_TERMINATE` in the ERISC's termination signal L1 address
   - Wait 1s for ERISC to exit cleanly
   - If still hung, request ERISC soft-reset via ARC message
   - After reset, re-initialize the fabric connection (full bringup sequence)
4. **Escalation**: If multiple ERISCs are hung (cluster-wide), this likely indicates a NOC deadlock or systemic issue. Full chip reset may be required.

### Heartbeat-Based Hang Location

For loops WITHOUT a WAYPOINT but WITH heartbeat updates, the host can detect the hang but not the specific loop. To improve this:
- Add a "last phase" register that the firmware updates before entering each major phase (init, handshake, main loop, teardown)
- Host reads this register when heartbeat stalls to narrow down the hang location

---

*End of Firmware Loop Census v1*
