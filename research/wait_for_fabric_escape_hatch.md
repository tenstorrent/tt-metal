<!--
SUMMARY: Analysis of escape hatch options for wait_for_fabric_endpoint_ready to prevent indefinite hangs when ERISC firmware fails to start
KEYWORDS: fabric, ERISC, hang, timeout, wait_for_fabric_endpoint_ready, dispatch, EDMStatus, escape hatch, race condition
SOURCE: Source code analysis of tt-metal worktree racecondition-main (April 2026)
SCOPE: Device-kernel-level and host-side mechanisms for preventing and detecting stuck fabric endpoints
USE WHEN: Designing or reviewing changes to fabric initialization, dispatch relay, or MUX startup timeout behavior
-->

# Escape Hatch for `wait_for_fabric_endpoint_ready`

## 1. The Spin Loop — What It Does Today

**File**: `tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp`

```cpp
FORCE_INLINE void wait_for_fabric_endpoint_ready(
    uint8_t fabric_ep_x, uint8_t fabric_ep_y,
    size_t fabric_ep_status_address,
    uint32_t local_fabric_ep_status_address,
    uint32_t max_poll_iterations = 1'000'000) {
    uint64_t noc_addr = get_noc_addr(fabric_ep_x, fabric_ep_y, fabric_ep_status_address);
    auto local_fabric_ep_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_fabric_ep_status_address);

    local_fabric_ep_status_ptr[0] = tt::tt_fabric::FabricEndpointStatus::TERMINATED;
    for (uint32_t i = 0; i < max_poll_iterations; ++i) {
        noc_async_read_one_packet(noc_addr, local_fabric_ep_status_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (local_fabric_ep_status_ptr[0] == tt::tt_fabric::FabricEndpointStatus::READY_FOR_TRAFFIC) {
            return;
        }
    }
    // Fall through on timeout
}
```

**Current state**: Already has a bounded loop with `max_poll_iterations = 1'000'000`. The companion function `wait_for_fabric_endpoint_terminated` uses the same pattern. Both fall through on timeout.

**What it polls**: Reads the remote core's `edm_status_address` via a NOC one-packet read, looking for `EDMStatus::READY_FOR_TRAFFIC` (0xA3B3C3D3).

**However**, an older *unbounded* variant exists: `wait_for_mux_endpoint_ready` in `fabric_router_udm_mux_extension.cpp` and `fabric_router_relay_extension.cpp`:

```cpp
FORCE_INLINE void wait_for_mux_endpoint_ready(...) {
    // ...
    do {
        noc_async_read_one_packet(noc_addr, mux_status_readback_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
    } while (ptr[0] != FabricMuxStatus::READY_FOR_TRAFFIC);
    // ^^^ NO TIMEOUT — infinite loop
}
```

This is a separate function, called by the ERISC router to wait for downstream Tensix MUX cores, and has **no escape hatch at all**.


## 2. EDMStatus Enum — All States

**File**: `tt_metal/fabric/fabric_edm_packet_header.hpp`

```
Value                          Hex          Meaning
STARTED                        0xA0B0C0D0   Kernel began executing
REMOTE_HANDSHAKE_COMPLETE      0xA1B1C1D1   Handshake with remote peer done
LOCAL_HANDSHAKE_COMPLETE       0xA2B2C2D2   Local handshake done (what host sync polls)
READY_FOR_TRAFFIC              0xA3B3C3D3   Fully ready (what device kernels poll)
TERMINATED                     0xA4B4C4D4   Kernel exiting
INITIALIZATION_STARTED         0xB0C0D0E0   Init postcodes (progressive states)
TXQ_INITIALIZED                0xB1C1D1E1
STREAM_REG_INITIALIZED         0xB2C2D2E2
DOWNSTREAM_EDM_SETUP_STARTED   0xB3C3D3E3
EDM_VCS_SETUP_COMPLETE         0xB4C4D4E4
WORKER_INTERFACES_INITIALIZED  0xB6C6D6E6
ETHERNET_HANDSHAKE_COMPLETE    0xB7C7D7E7
VCS_OPENED                     0xB8C8D8E8
ROUTING_TABLE_INITIALIZED      0xB9C9D9E9
INITIALIZATION_COMPLETE        0xBACADAEA
```

**No FAILED/ERROR state exists.** There is no way for the ERISC to signal "I tried to start but couldn't." If ERISC never starts, the status address retains whatever was last written (often 0 after `configure_fabric_cores()` clears it, or stale values like 0x49705180 from a prior crash).


## 3. Call Sites of `wait_for_fabric_endpoint_ready`

### 3a. `cq_relay.hpp` (Dispatch Relay Kernel — Tensix core)
- Called inside `CQRelayClient::init()` when `FABRIC_RELAY` is defined.
- After the wait returns, calls `fabric_client_connect<>()` (opens EDM connection).
- **No error handling after the wait**. If it falls through on timeout, the connect and subsequent fabric writes will operate on an uninitialized endpoint.

### 3b. `fabric_router_mux_extension.cpp` (Tensix MUX kernel)
- Called at startup, waits for the ERISC fabric router to be ready.
- After the wait, opens the fabric connection and sets `status_ptr[0] = READY_FOR_TRAFFIC`.
- If the wait falls through, the MUX marks itself READY_FOR_TRAFFIC even though its upstream ERISC isn't ready — propagating the problem.

### 3c. `fabric_router_udm_mux_extension.cpp` (UDM MUX kernel)
- Same pattern as 3b: waits for ERISC, opens connection, sets READY_FOR_TRAFFIC.
- Also calls `wait_for_mux_endpoint_ready()` (the unbounded variant) for downstream MUX connections.

### 3d. `tt_fabric_mux.cpp` (Standalone MUX kernel)
- Same pattern: wait for ERISC, open connection, set READY_FOR_TRAFFIC.


## 4. Host-Side Post-`configure_fabric_cores()` Polling

### What `configure_fabric_cores()` does (fabric_init.cpp)
Clears L1 addresses on all active ETH router channels. That's it — no polling, no readiness check.

### What happens after it
The call chain is:
1. `FabricFirmwareInitializer::compile_and_configure_fabric()` — compiles and launches firmware
2. `FabricFirmwareInitializer::configure()` — calls `wait_for_fabric_router_sync()` then `verify_all_fabric_channels_healthy()`

### `wait_for_fabric_router_sync()` (fabric_firmware_initializer.cpp:597)
- Polls the master ERISC channel's `edm_status_address` for `LOCAL_HANDSHAKE_COMPLETE` (0xA2B2C2D2).
- **Bounded by `timeout_ms`** (configurable, default via `get_fabric_router_sync_timeout_ms()`).
- Throws `TT_THROW` on timeout.
- After sync succeeds, if `get_fabric_router_ready_address_and_signal()` returns a value, it **writes** `READY_FOR_TRAFFIC` to the master router's status address (a "go signal").

### `verify_all_fabric_channels_healthy()` (fabric_firmware_initializer.cpp:676)
- Runs **after** `wait_for_fabric_router_sync()`.
- Checks ALL active ERISC channels (not just master) for `READY_FOR_TRAFFIC`.
- Retries up to 3 times with 10ms delay.
- Distinguishes corrupt channels (unrecognized status) from still-initializing channels (valid EDMStatus).
- **Throws on failure** — fail-fast before dispatch starts.

### Bottom line
There IS already a host-side check that all ERISCs are READY_FOR_TRAFFIC before dispatch begins. The problem is when an ERISC becomes stuck **after** this check passes (e.g., during `quiesce_and_restart_fabric_workers` re-initialization).


## 5. Host-Side Timeout Mechanism

### `TT_METAL_OPERATION_TIMEOUT_SECONDS`
- Default: **0.0 (no timeout)**. CI typically sets this to 5s.
- Applied via `loop_and_wait_with_timeout()` in `system_memory_manager.cpp`.
- On timeout, calls `MetalContext::on_dispatch_timeout_detected()` which logs errors, optionally runs tt-triage, and then throws.
- **No kernel abort is triggered.** The host throws an exception; the device kernel continues spinning.

### `quiesce_and_restart_fabric_workers` Phase 4 (device.cpp:777)
- Already has a host-side bounded poll (5000ms) for MUX READY_FOR_TRAFFIC.
- On timeout: **force-resets the Tensix MUX core** via `assert_risc_reset_at_core(ALL)`.
- This is the recovery mechanism for a MUX that fails to start (because its ERISC is stuck).

### `teardown_fabric_config` (metal_env.cpp:271)
- Polls all ERISC channels for `TERMINATED` status with 5s timeout.
- On timeout: currently **skips** force-reset (experimental F5a flag to avoid cross-chip corruption).


## 6. Device Kernel Timeout Mechanisms Available

### What exists on Tensix/ERISC cores
1. **Bounded iteration loops** — The approach already used by `wait_for_fabric_endpoint_ready` and `wait_for_fabric_endpoint_terminated`. No hardware timer, just a counter.
2. **RISC_POST_HEARTBEAT** — Writes an incrementing counter to address 0x1C (Wormhole only). Used by idle ERISC dispatch kernels. The host watcher can detect a stuck heartbeat, but this is a monitoring mechanism, not a timeout trigger.
3. **WAYPOINT markers** — Debug breadcrumbs (e.g., `WAYPOINT("FMCW")` before the wait, `WAYPOINT("FMCD")` after). Used by the watcher for diagnostics but not for timeout control.

### What does NOT exist
- **No hardware watchdog timer** accessible to kernels. There's no "set a timer, get an interrupt" primitive.
- **No host-to-device abort signal** that a spinning kernel checks. The host can force-reset the core (killing it), but can't politely ask it to stop.
- **No completion event for failure**. Kernels can only signal success (via completion queue writes). There's no "I failed" path back to the host.


## 7. The Remaining Unbounded Spin: `wait_for_mux_endpoint_ready`

This is the **most dangerous remaining unbounded spin**. It exists in two files:
- `fabric_router_udm_mux_extension.cpp:192`
- `fabric_router_relay_extension.cpp:156`

Both use `do { ... } while (ptr[0] != READY_FOR_TRAFFIC)` with no iteration bound. If a downstream Tensix MUX core fails to start, the ERISC router calling this function will spin forever.

This should be converted to the same bounded-loop pattern as `wait_for_fabric_endpoint_ready`.


## 8. Recommended Changes (Smallest/Safest Escape Hatch)

### Change 1: Bound `wait_for_mux_endpoint_ready` (CRITICAL)
Convert the unbounded `do/while` to a bounded `for` loop matching `wait_for_fabric_endpoint_ready`:

```cpp
FORCE_INLINE void wait_for_mux_endpoint_ready(
    uint8_t mux_noc_x, uint8_t mux_noc_y,
    size_t mux_status_address, uint32_t mux_status_readback_address,
    uint32_t max_poll_iterations = 1'000'000) {
    uint64_t noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, mux_status_address);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mux_status_readback_address);
    ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
    for (uint32_t i = 0; i < max_poll_iterations; ++i) {
        noc_async_read_one_packet(noc_addr, mux_status_readback_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (ptr[0] == tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC) {
            return;
        }
    }
    // Fall through on timeout
}
```

**Risk**: Low. The 1M iteration default (~seconds at NOC latencies) is conservative. A healthy MUX starts well within this bound. The host-side 5s timeout in `completion_queue_wait_front` will catch and report the failure.

### Change 2: Don't propagate READY_FOR_TRAFFIC on timeout
In MUX kernels (`tt_fabric_mux.cpp`, `fabric_router_mux_extension.cpp`, `fabric_router_udm_mux_extension.cpp`), check the return status before writing `READY_FOR_TRAFFIC`:

Currently:
```cpp
wait_for_fabric_endpoint_ready(x, y, status_addr, local_addr);
fabric_connection.open<...>();
status_ptr[0] = FabricMuxStatus::READY_FOR_TRAFFIC;  // Always writes this
```

Ideal:
```cpp
wait_for_fabric_endpoint_ready(x, y, status_addr, local_addr);
if (local_fabric_ep_status_ptr[0] == READY_FOR_TRAFFIC) {
    fabric_connection.open<...>();
    status_ptr[0] = FabricMuxStatus::READY_FOR_TRAFFIC;
} else {
    status_ptr[0] = FabricMuxStatus::TERMINATED;
    // Skip main loop, let host-side timeout detect and reset
    return;
}
```

**Risk**: Medium. Changing the MUX early-exit path needs careful testing. The host's Phase 4 poll will see TERMINATED instead of READY_FOR_TRAFFIC, and the force-reset path already handles this.

### Change 3: Make `wait_for_fabric_endpoint_ready` return success/failure
Currently it returns `void`. Change to `bool` so callers can branch:

```cpp
FORCE_INLINE bool wait_for_fabric_endpoint_ready(...) {
    // ... same loop ...
    if (local_fabric_ep_status_ptr[0] == READY_FOR_TRAFFIC) return true;
    // ...
    return false;  // timeout
}
```

**Risk**: Low. All existing callers currently ignore the return value (void), so adding a bool return is backward-compatible. Callers can be updated incrementally.


## 9. Ideal Architecture

```
Layer           Mechanism                           Today            Ideal
─────────────────────────────────────────────────────────────────────────────────
Device kernel   Bounded poll in wait_for_*_ready    ✅ (1M iters)    ✅ + return bool
                                                    ❌ wait_for_mux   ✅ bounded
                Propagate READY only on success     ❌ always writes  ✅ conditional
                Write FAILED status on timeout      ❌ no such state  ✅ add EDMStatus::FAILED

Host pre-       verify_all_fabric_channels_healthy  ✅ exists         ✅ (already good)
dispatch

Host Phase 4    Poll MUX for READY_FOR_TRAFFIC      ✅ 5s timeout     ✅ + detect FAILED
(quiesce)       Force-reset on timeout              ✅ exists         ✅ (already good)

Host runtime    completion_queue_wait_front          ✅ 5s timeout     ✅ (already good)
                on_dispatch_timeout_detected         ✅ logs/triage    ✅ (already good)
```

The key gap is at the device-kernel level: the unbounded `wait_for_mux_endpoint_ready` and the unconditional `READY_FOR_TRAFFIC` write. Fixing these two issues means a stuck ERISC or MUX will:
1. Not wedge any downstream kernel (bounded poll falls through)
2. Not falsely signal readiness (conditional READY_FOR_TRAFFIC write)
3. Be caught by the existing host-side timeout + force-reset machinery

Adding `EDMStatus::FAILED` would improve diagnostics but is not strictly required for the escape hatch — the host can already distinguish "stuck at 0x0" from "at READY_FOR_TRAFFIC" by reading the status word.
