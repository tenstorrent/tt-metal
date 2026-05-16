<!--
SUMMARY: Race conditions and deadlock findings in EDM fabric / ETH / ERISC device-side code
KEYWORDS: race-condition, deadlock, edm, fabric, erisc, brisc, handshake, semaphore
SOURCE: Local code analysis of tt-metal nsexton/0-racecondition-hunt branch
SCOPE: All EDM handshake, datamover, firmware init/teardown, and L1 semaphore code
USE WHEN: Debugging fabric hangs, EDM deadlocks, or ERISC initialization failures
-->

# EDM Race Condition Findings

## Hazard 1: fabric_sender_side_handshake() missing FIX HS1 final send
- **Severity**: CRITICAL
- **Location**: tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp:22-50
- **Mechanism**: `fabric_sender_side_handshake()` is the fabric-specific variant of `sender_side_handshake()`. FIX HS1 added a post-loop final `eth_send_packet()` to `sender_side_handshake()` to handle the simultaneous-sender race (both sides call sender, one exits early after seeing MAGIC but the other's init_handshake_info() erased its earlier sends). The fabric variant was NOT updated — it exits the loop at line 49 with no final send, leaving the remote stuck if the simultaneous-sender scenario occurs on a fabric handshake path.
- **Trigger conditions**: Two devices initialize fabric channels concurrently via `fabric_sender_side_handshake()`. One device's UMD relay is faster, creating the timing window for the init_handshake_info() reset to erase pre-loop sends.
- **Fix**: Add post-loop final `eth_send_packet()` to `fabric_sender_side_handshake()`, guarded by termination signal check (matching the Blackhole non-ARCH_WORMHOLE pattern).
- **Commit**: FIX HS2 — applied and pushed.

---

## Areas Investigated — No New Hazards Found

### Area 2: EDM Worker Datamover State Machine
**Files**: `fabric_erisc_datamover_channels.hpp`, `edm_fabric_flow_control_helpers.hpp`, `fabric_router_flow_control.hpp`, `fabric_erisc_router_speedy_path.hpp`
- Credit system uses monotonically increasing `ChannelCounter` — no ABA problem possible.
- Ring buffer indices wrapped via power-of-2 `wrap_increment()` — correct.
- Single-owner pattern: only one side writes each buffer index (sender writes wr, receiver writes rd).
- Speedy path pre-send teardown check (line 124) is non-looping single check — stale read at most delays teardown by one iteration, not a deadlock.
- Counter-based credit sender uses `invalidate_l1_cache()` before reading remote counters.

### Area 3: ERISC Main Loop and Context Switch
**Files**: `erisc.cc` (WH), `active_erisc.cc` (BH), `tunneling.h`
- Go-signal polling uses `invalidate_l1_cache()` before each read.
- `run_routing()` is cooperative context switch (`risc_context_switch()`).
- Blackhole two-ERISC mode: ERISC0 launches ERISC1, then enters reset/resume cycle — no shared-state race.
- No missed-wake signals: polling is continuous (no wait/notify pattern).

### Area 4: fabric_firmware_initializer.cpp Launch and Teardown
**File**: `fabric_firmware_initializer.cpp` (3002 lines)
- Massively hardened by dozens of prior fixes (FIX A through FIX TJ).
- Two-phase probe-then-configure (FIX J) prevents TOCTOU.
- Non-MMIO-first ordering (FIX J2) prevents relay races.
- `terminate_stale_erisc_routers()` probes channels with retry + timeout.
- No new hazards found.

### Area 5: device.cpp Phase Ordering and Barrier Races
**File**: `device.cpp` — `wait_for_fabric_workers_ready()` and `quiesce_and_restart_fabric_workers()`
- Concurrent `wait_for_fabric_workers_ready()` on different devices is NOT a concern — calls are sequential (mesh-level caller iterates one device at a time).
- Phase 5 relay read failures set `fabric_relay_path_broken_` atomically (`std::atomic<bool>`) — persists across quiesce calls.
- Phase 5b has per-read deadline guard preventing timeout accumulation.
- Phase 2.5 ERISC termination: send TERMINATE, poll with timeout, force-reset on failure.
- Phase 3 skips soft reset for all channels (FIX N/AD) — preserves relay path.
- Three-pass quiesce (1a/1b/1c) with explicit STARTED barrier between non-MMIO devices (FIX AE/AF) eliminates simultaneous-handshake deadlock.
- `phase5b_erisc_health_check()` distinguishes pre-dead, partial-mesh, and truly-unexpected states.
- No new hazards found.

### Area 6: BRISC to ERISC Shared L1 Semaphores
**Files**: `edm_fabric_worker_adapters.hpp`, `fabric_erisc_datamover_channels.hpp`, `tt_fabric_utils.h`, `tt_fabric_mux.cpp`
- Worker-to-ERISC communication uses NOC writes (atomic at 32-bit granularity).
- ERISC-to-worker uses `noc_semaphore_inc` (atomic increment).
- `connection_live_semaphore` is `volatile tt_l1_ptr` — correct for WH (no CPU data cache on ERISC).
- MUX kernel calls `invalidate_l1_cache()` unconditionally before `check_worker_connections()`.
- `close_finish()` polling loop has `invalidate_l1_cache()` per iteration.
- `teardown_worker_connection()` calls `router_invalidate_l1_cache()` before reading worker info.
- No critical races found.

### Area 7: Non-MMIO UMD Relay Ordering
**Files**: `mesh_device.cpp` (quiesce_internal, restart_fabric_workers_for_quiesce, get_fabric_quiesce_restart_order)
- Three-pass quiesce approach: 1a (all devices setup, ETH deferred), 1b (MMIO ETH launch), 1c (non-MMIO ETH launch with STARTED barrier per device).
- Non-MMIO-first ordering in all passes prevents relay disruption.
- Tunnel-order readiness polling (farthest-to-closest) matches initial startup sequence.
- FIX AF explicit STARTED polling eliminates relay-latency assumption that caused simultaneous-handshake deadlock.
- No new hazards found.

---

## Summary

**Total hazards found**: 1
- Hazard 1 (FIX HS2): CRITICAL — `fabric_sender_side_handshake()` missing post-loop final send. Fixed and pushed.

**Areas cleared (no hazards)**: 6 of 7 investigation areas
- The codebase is heavily hardened with dozens of prior fixes (FIX A through FIX TJ, FIX HS1, FIX RZ4, etc.)
- Credit/flow-control system is well-designed with monotonic counters and single-owner patterns
- Cache invalidation is properly applied at all shared-data read points
- Quiesce ordering is carefully sequenced to prevent relay races and simultaneous-handshake deadlocks
