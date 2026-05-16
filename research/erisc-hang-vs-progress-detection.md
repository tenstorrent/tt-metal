<!--
SUMMARY: How to distinguish a real ERISC hang from a legitimate long-running fabric operation — current timeout mechanisms, internal progress indicators, and proposed approaches.
KEYWORDS: ERISC, hang detection, fabric, EDM, timeout, heartbeat, progress, telemetry, dispatch, watchdog, liveness
SOURCE: Code analysis of tt-metal racecondition-fix-ae worktree (May 2026)
SCOPE: Fabric router kernel internals, host-side polling, dispatch timeout, telemetry heartbeats
USE WHEN: Designing progress-aware timeout for fabric/dispatch, debugging ERISC hangs, improving hang detection
-->

# Distinguishing Real ERISC Hangs from Legitimate Long-Running Operations

## Problem Statement

The current `TT_METAL_OPERATION_TIMEOUT_SECONDS` mechanism uses a flat wall-clock timeout (default: unset / 0 = infinite; CI typically sets 5s). If a CCL op (AllGather, ReduceScatter) legitimately takes >5s on large tensors, the host fires a timeout and declares a hang. Conversely, if an ERISC is genuinely stuck (spinning on a broken channel, waiting for a credit that will never arrive), the only signal is the same wall-clock timeout.

**The core question**: what internal ERISC state can the host observe to distinguish "making progress, just slow" from "stuck in an infinite loop"?

---

## 1. Current Timeout Mechanisms

### 1.1 Operation Timeout (`TT_METAL_OPERATION_TIMEOUT_SECONDS`)

**Config**: `tt_metal/llrt/rtoptions.cpp:975` — env var `TT_METAL_OPERATION_TIMEOUT_SECONDS`, parsed as float seconds.

**Where it fires**: `tt_metal/impl/dispatch/system_memory_manager.cpp:56-106` — the `loop_and_wait_with_timeout()` template.

**What it polls**: Two callsites:
1. **fetch_queue_reserve_back** (line 829) — waiting for prefetch queue space
2. **completion_queue_wait_front** (line 878) — waiting for dispatch to write a completion

Both call `get_cq_dispatch_progress()` (line 825, 874) which reads a 32-bit counter from the dispatch kernel's L1 memory.

**Progress-aware timeout logic** (lines 62-106):
```
last_progress_time = now()
while wait_condition():
    if (now - last_progress_update_time >= progress_update_interval):
        current_progress = get_progress()           // reads L1 counter
        if current_progress != last_progress_value:
            last_progress_time = now()               // reset timeout clock
    if (now - last_progress_time >= timeout_duration):
        on_timeout()                                 // THROW
```

**Key insight**: The timeout is already progress-aware for DISPATCH kernels (Tensix dispatch). Every time the dispatch kernel processes a command, it increments `dispatch_progress` at `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1475`. The host reads this counter every `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS` (default 100ms). If the counter advances, the timeout clock resets.

**The gap**: This progress counter only tracks the DISPATCH kernel (Tensix), not the fabric ERISC router. When an op is waiting on fabric (e.g., AllGather blocked on data flowing through EDM), the dispatch kernel has already issued the command and is blocked in `wait_for_available_data_and_release_old_pages()` (line 1467). The dispatch_progress counter stops advancing. From the host's perspective, there is no progress — indistinguishable from a hang.

### 1.2 Fabric Router Sync Timeout

**Where**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:1881-1998`

**What**: `wait_for_fabric_router_sync()` polls the EDMStatus field at `router_sync_address` on each device's master router channel. It waits for `LOCAL_HANDSHAKE_COMPLETE` (or similar expected_status).

**Timeout**: Default 10000ms (line 2273), configurable via `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS`.

**This is initialization-only** — it runs during `FabricFirmwareInitializer::configure()` and has nothing to do with runtime hang detection.

### 1.3 ETH Heartbeat (Base Firmware)

**Where**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:380-510`

**What**: During teardown/reset, the host polls a heartbeat address:
- WH: `0x1F80` (test_results[48])
- BH: `0x7CC70` (MEM_SYSENG_ETH_HEARTBEAT)

This is written by the **base UMD relay firmware** (not the fabric router). It proves the ERISC ROM booted and base firmware is alive. **Not useful for runtime progress detection** — it only confirms the ERISC is alive at the firmware level, not that the fabric router kernel is making progress.

---

## 2. ERISC Internal State That Advances During Work

### 2.1 Fabric Kernel Heartbeat Counter (EXISTING)

**File**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:2264-2270, 2619-2621`

```cpp
uint16_t fabric_heartbeat_counter = 0;
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70; // BH
// or 0x1F80 for WH
volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);

// In main loop, every 64 iterations:
if ((++fabric_heartbeat_counter & 0x3F) == 0) {
    *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
}
```

**What it proves**: The ERISC main loop is iterating. It writes `0xDCBA<counter>` to a well-known L1 address every 64 loop iterations.

**Limitation**: This advances even if the router is idle (no packets to process). It proves "not crashed" but NOT "making progress on work." A router spinning on an empty channel still increments this counter. However, a router stuck in an inner spin-wait (e.g., `eth_txq_is_busy()` spin or a `wait_for_notification()` call) would NOT increment it, since those spin inside helper functions without returning to the main loop.

**Host readability**: The host can read this address via `read_core()` / `cluster_.read_reg()`. The address is the same as the base firmware heartbeat address, so the infrastructure for reading it already exists.

### 2.2 Telemetry Heartbeats — TX and RX (EXISTING, compile-time gated)

**File**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:1526-1548`

**Structures**: `tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h:54-60`

```cpp
struct EriscDynamicEntry {
    RouterState router_state;          // INITIALIZING/RUNNING/PAUSED/DRAINING/RETRAINING
    RiscTimestampV2 tx_heartbeat;      // 64-bit counter
    RiscTimestampV2 rx_heartbeat;      // 64-bit counter
};
```

**TX heartbeat** increments when:
- `tx_progress == true` (a packet was sent over Ethernet), OR
- All sender queues are empty (`sender_idle` — checked via `any_sender_channels_active()`)

**RX heartbeat** increments when:
- `rx_progress == true` (a packet was forwarded from receiver to NoC/local), OR
- Receiver queues are empty (`get_ptr_val<to_receiver_pkts_sent_id>() == 0`)

**Key insight for hang detection**: If both TX and RX heartbeats stop advancing AND the router is NOT idle (sender channels have pending data), that means the ERISC is stuck — it has work to do but cannot make progress. This is the **exact signal** we need.

**Current limitation**: These heartbeats are compile-time gated by `FABRIC_TELEMETRY_HEARTBEAT_TX` and `FABRIC_TELEMETRY_HEARTBEAT_RX`, which are part of the `FABRIC_TELEMETRY_STATS_MASK`. They are stored in the `FabricTelemetry` struct in L1. **Whether they are currently enabled by default in production builds needs to be verified** — the mask is set via `NAMED_CT_ARG("FABRIC_TELEMETRY_STATS_MASK")` at compile time.

### 2.3 Bandwidth Telemetry Counters (EXISTING, compile-time gated)

**File**: `fabric_erisc_router.cpp:1602-1614`, `fabric_telemetry_msgs.h:45-52`

```cpp
struct BandwidthTelemetry {
    RiscTimestampV2 elapsed_active_cycles;   // cycles where work was done
    RiscTimestampV2 elapsed_cycles;          // total cycles (active + idle)
    uint64_t num_words_sent;                 // free-running word counter
    uint64_t num_packets_sent;               // free-running packet counter
};
```

Both `tx_bandwidth` and `rx_bandwidth` are maintained. `num_packets_sent` and `num_words_sent` advance monotonically whenever a packet is sent or forwarded.

**Usefulness**: `num_packets_sent` is a strong progress indicator — if it advances between two host reads, the ERISC is doing real work. Gated by `FABRIC_TELEMETRY_BANDWIDTH`.

### 2.4 Sender Channel Free Slots (Stream Registers)

**File**: `fabric_erisc_router.cpp:1682-1683`

```cpp
uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
```

Each sender channel maintains a `free_slots` counter in a hardware stream register. When `free_slots < num_buffers`, there is pending data from workers waiting to be sent. The `any_sender_channels_active()` function (line 1411) iterates all sender channels and returns true if any have unsent data.

**Host readability**: Stream registers are readable via NOC reads to L1, but the stream register addresses are derived from compile-time args and not directly exposed to host code.

### 2.5 Receiver Packets Pending (Stream Registers)

**File**: `fabric_erisc_router.cpp:1821`

```cpp
auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();
```

This stream register tracks how many packets have been received from Ethernet but not yet processed/forwarded. Non-zero means the receiver has work to do.

### 2.6 Credit Counters (ack / completion)

**File**: `fabric_erisc_router_ct_args.hpp:465-486`

```
to_sender_remote_ack_counters_base_address
to_sender_remote_completion_counters_base_address
local_receiver_ack_counters_base_address
local_receiver_completion_counters_base_address
```

These are L1 memory addresses holding unbounded credit counters. They advance as packets are acknowledged and completed. The host can read these addresses to determine if credits are flowing.

### 2.7 EDMStatus (Static State)

**File**: `tt_metal/fabric/fabric_edm_packet_header.hpp:48-94`

The `EDMStatus` enum (e.g., `READY_FOR_TRAFFIC = 0xA3B3C3D3`) is a coarse lifecycle indicator. It does NOT change during normal data flow — once the router reaches `READY_FOR_TRAFFIC`, it stays there until `TERMINATED`. **Not useful for progress detection**, only for initialization/teardown state.

### 2.8 RouterState (Dynamic, in FabricTelemetry)

**File**: `fabric_telemetry_msgs.h:15`, written at `fabric_erisc_router.cpp:1524`

```cpp
enum class RouterState : uint32_t { INITIALIZING=0, RUNNING=1, PAUSED=2, DRAINING=3, RETRAINING=4 };
```

Updated in the telemetry struct every loop iteration (when `FABRIC_TELEMETRY_ROUTER_STATE` is enabled). Useful for distinguishing "router is paused/retraining" from "router is running but stuck."

### 2.9 `did_something` / `did_nothing_count` (Local Only)

**File**: `fabric_erisc_router.cpp:515, 2626-2639`

```cpp
bool did_something;  // global, set to true when any channel makes progress
// ...
if (did_something) {
    did_nothing_count = 0;
} else {
    if (did_nothing_count++ > SWITCH_INTERVAL) {
        // context switch to base firmware
    }
}
```

This is the router's own idle detection. After `SWITCH_INTERVAL` iterations with no progress, it context-switches to base firmware (which handles link training, etc.). **Not directly host-readable** — it's a local variable on the ERISC stack.

---

## 3. Distinguishable States

### State Matrix

```
State                           Heartbeat   TX/RX HB    Packets     Meaning
─────────────────────────────   ─────────   ─────────   ─────────   ─────────────────────────
Idle (no work pending)          advancing   advancing*  no change   Normal — nothing to do
Active (sending/receiving)      advancing   advancing   advancing   Normal — work in progress
Waiting for downstream credit   advancing   STALLED     no change   Blocked but may be legit
Hung (broken channel)           advancing** STALLED     no change   REAL HANG
Crashed (ERISC firmware dead)   STOPPED     STOPPED     no change   Severe failure
Stuck in eth_txq spin           STOPPED***  STOPPED     no change   May or may not be hang
```

`*` TX/RX heartbeats advance even when idle (they explicitly check `sender_idle` / `receiver_idle`)
`**` Main loop heartbeat (`0xDCBA...`) advances even when no packets flow, as long as the main loop runs
`***` If the ERISC is stuck inside `eth_txq_is_busy()` spin-wait (line 1686-1688), the main loop heartbeat does NOT advance because the spin is inside the sender step function

### Key Distinction

The critical case is **"Waiting for downstream credit"** vs **"Hung"**. Both look identical from a single snapshot:
- Heartbeat advancing (main loop alive)
- No packets flowing
- Sender channels have pending data

**To distinguish them**, you need **temporal** observation:
1. Read credit counters at time T1
2. Wait N seconds
3. Read credit counters at time T2
4. If credits advanced: progress (waiting was legitimate, downstream was just slow)
5. If credits unchanged AND sender has data AND receiver has no data: HUNG

---

## 4. Proposed Mechanisms

### 4.1 Approach A: Expose Fabric Heartbeats to `loop_and_wait_with_timeout()` (Recommended)

**Concept**: Extend the existing `get_cq_dispatch_progress()` to also read the fabric router's telemetry heartbeat counters. Combine dispatch progress + fabric progress into a single 32-bit progress value.

**Implementation**:

1. **Kernel side**: Ensure `FABRIC_TELEMETRY_HEARTBEAT_TX` and `FABRIC_TELEMETRY_HEARTBEAT_RX` are always enabled (remove compile-time gating or set the mask to always include them). Alternatively, write a simpler always-on progress counter to a dedicated L1 address (like `fabric_heartbeat_counter` but with a semantic of "total packets processed").

2. **Host side**: In `get_cq_dispatch_progress()` (`tt_metal/impl/dispatch/command_queue_common.cpp:128`), after reading the dispatch progress, also read the fabric telemetry heartbeat from each active ERISC. XOR or hash them together with dispatch_progress to produce a combined progress value.

3. **Timeout behavior**: If either dispatch or fabric is making progress, the timeout clock resets. The timeout only fires when BOTH are stalled.

**Tradeoffs**:
- (+) Minimal kernel changes — heartbeats already exist
- (+) Fits cleanly into existing timeout framework
- (-) Additional PCIe reads per progress check (one per ERISC) — but progress is only checked every 100ms, so overhead is negligible
- (-) Telemetry heartbeats currently compile-time gated — need to either always-enable or add a simpler always-on counter

**Difficulty**: LOW. ~50 lines of host code change, 0-10 lines of kernel code change (if heartbeats already enabled).

### 4.2 Approach B: Dedicated Always-On Packet Counter at Well-Known L1 Address

**Concept**: Add a single `uint32_t total_packets_processed` counter to a well-known L1 address in the fabric router. Increment it on every packet sent or received. No compile-time gating.

**Implementation**:

1. **Kernel side**: In `run_sender_channel_step_impl()` after `progress = true` (line 1691), increment a counter at a fixed L1 address. Similarly in `run_receiver_channel_step_impl()` after `progress = true` (line 1913). Total: 2 lines of kernel code.

2. **Host side**: Read this address in `get_cq_dispatch_progress()` or a new `get_fabric_progress()` function.

**Tradeoffs**:
- (+) Zero compile-time gating — always available
- (+) Single L1 read per ERISC — minimal overhead
- (+) Semantically clean: "packets processed" = work done
- (-) One more L1 write per packet on the critical path (negligible: single 32-bit store to L1 is <10 cycles)
- (-) Need to pick a well-known L1 address and wire it through compile-time args

**Difficulty**: LOW. ~10 lines of kernel code, ~30 lines of host code.

### 4.3 Approach C: Multi-Level Liveness with Credit Flow Analysis

**Concept**: On timeout, instead of immediately throwing, enter a "diagnostic probe" mode: read ALL credit counters, free-slot registers, and packet counters from every ERISC. Determine if any channel is making progress. Only throw if all channels are completely stalled.

**Implementation**:

1. **On timeout trigger**: Instead of `on_timeout()` throwing immediately, call a diagnostic function.
2. **Diagnostic function**: For each device, for each active ERISC:
   - Read `fabric_heartbeat_ptr` (main loop alive?)
   - Read `to_sender_remote_ack_counters_base_address` (acks flowing?)
   - Read `to_sender_remote_completion_counters_base_address` (completions flowing?)
   - Read sender channel free slots (via stream reg L1 mirror) (data pending?)
   - Compare with snapshot from N seconds ago
3. **Decision**: If any counter advanced, extend timeout. If all stalled + data pending, declare hang.

**Tradeoffs**:
- (+) Most accurate — can pinpoint exactly which channel is stuck
- (+) Rich diagnostic output for post-mortem
- (-) Complex: need to know L1 layout of all credit counters per ERISC
- (-) Multiple PCIe reads per ERISC per diagnostic pass
- (-) Need compile-time arg addresses to be available at host runtime (they are, in the builder context)

**Difficulty**: MEDIUM. ~200 lines of host code, 0 kernel code changes. Needs careful plumbing of ERISC L1 addresses from builder context to the diagnostic function.

### 4.4 Approach D: Reuse Existing `fabric_heartbeat_counter` (Simplest)

**Concept**: The `0xDCBA<counter>` value at `FABRIC_KERNEL_HEARTBEAT_ADDR` already proves main-loop liveness. Simply read this in `get_cq_dispatch_progress()`.

**Implementation**:

1. The address is already well-known: `0x7CC70` (BH) / `0x1F80` (WH).
2. In `get_cq_dispatch_progress()`, read this address for each active ERISC.
3. XOR into the progress value.

**Tradeoffs**:
- (+) Zero kernel changes — already written every 64 loop iterations
- (+) Same address as base firmware heartbeat — host code already knows how to read it
- (-) Does NOT distinguish "idle but alive" from "has work but stuck" — it advances even when the router is idle
- (-) Only proves the main loop is running, not that work is being done

**Difficulty**: VERY LOW. ~20 lines of host code.

**This is the weakest approach** — it prevents timeout on idle-but-alive ERISCs, but cannot detect a hang where the main loop runs but no packets flow (e.g., stuck waiting for credit from a dead downstream).

---

## 5. Implementation Difficulty Summary

```
Approach   Kernel Changes   Host Changes   Accuracy    Difficulty
─────────  ──────────────   ────────────   ─────────   ──────────
A (HB)     0-10 lines       ~50 lines      HIGH        LOW
B (pkt)    ~10 lines        ~30 lines      HIGH        LOW
C (diag)   0 lines          ~200 lines     HIGHEST     MEDIUM
D (basic)  0 lines          ~20 lines      LOW         VERY LOW
```

### Recommended Path

**Start with Approach B** (dedicated packet counter): simplest, always-on, no compile-time gating, and semantically meaningful. It answers the exact question: "is this ERISC processing packets?"

**Then layer Approach C** as a diagnostic enhancement: when the timeout fires despite the packet counter being stalled, dump the full credit/slot state for forensic analysis. This gives ops engineers everything they need to identify the stuck channel.

### What Needs to Change

**Kernel (`fabric_erisc_router.cpp`)**:
1. Add a `uint32_t` at a well-known L1 address (e.g., `FABRIC_KERNEL_HEARTBEAT_ADDR + 4`)
2. Increment in `run_sender_channel_step_impl()` when `progress = true`
3. Increment in `run_receiver_channel_step_impl()` when `progress = true`

**Host (`command_queue_common.cpp`)**:
1. Extend `get_cq_dispatch_progress()` to also read fabric packet counters
2. Combine into a single progress indicator

**Host (`system_memory_manager.cpp`)**:
1. No changes — `loop_and_wait_with_timeout()` already handles progress-aware timeouts

---

## 6. Key File Reference

| File | What |
|------|------|
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Fabric router main loop, heartbeat, telemetry, sender/receiver steps |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | All compile-time args: channel addrs, telemetry config, credit counter addrs |
| `tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h` | FabricTelemetry struct definition (heartbeats, bandwidth, router state) |
| `tt_metal/fabric/fabric_edm_packet_header.hpp:48-94` | EDMStatus enum, TerminationSignal enum |
| `tt_metal/impl/device/edm_status_utils.hpp` | EDMStatus string helpers, EthDiagSentinel values |
| `tt_metal/fabric/hw/inc/tt_fabric_utils.h:24-31` | `got_immediate_termination_signal()` implementation |
| `tt_metal/impl/dispatch/system_memory_manager.cpp:56-106` | `loop_and_wait_with_timeout()` — the timeout framework |
| `tt_metal/impl/dispatch/command_queue_common.cpp:128-159` | `get_cq_dispatch_progress()` — dispatch progress counter reader |
| `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:1461-1476` | Dispatch kernel progress counter write |
| `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:1881-1998` | `wait_for_fabric_router_sync()` — init-time sync polling |
| `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:380-510` | Base firmware heartbeat polling (teardown) |
| `tt_metal/llrt/rtoptions.cpp:971-995` | Timeout env var parsing |
| `tt_metal/impl/context/metal_context.cpp:731-756` | `on_dispatch_timeout_detected()` — timeout action handler |
