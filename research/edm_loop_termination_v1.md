<!--
SUMMARY: Loop termination analysis — exit conditions, hang scenarios, and break-out mechanism proposals for EDM/ETH/ERISC/BRISC firmware
KEYWORDS: edm, fabric, erisc, hang, loop, termination, watchdog, timeout, wormhole
SOURCE: Serial adversarial audit of tt-metal branch nsexton/0-racecondition-hunt
SCOPE: 4 loop categories: wait_for_empty_write_slot, teardown drain, static-connection wait, WH-term-guard removal
USE WHEN: Investigating firmware hang with spinning/timeout symptoms in fabric bring-up or teardown
-->

# EDM Loop Termination Audit v1

Branch: `nsexton/0-racecondition-hunt`
Worktree: `/workspace/group/worktrees/nsexton-0-racecondition-hunt/`
Date: 2026-05-10

## Summary of Findings

4 categories of non-terminating loops were audited. All are real hang risks under specific failure conditions. Watchdog logging was added to all; no behavioral changes (the loops still spin, but now produce diagnosable WAYPOINT codes in watcher dumps).

## ITEM 1: `wait_for_empty_write_slot()` — edm_fabric_worker_adapters.hpp

**File**: `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:304`

**Exit condition**: `edm_has_space_for_packet<1>()` returns true — i.e., `get_num_free_write_slots() >= 1`.

**Who sets it**: The downstream EDM processes the packet and either:
- (non-stream path) increments `*edm_buffer_local_free_slots_read_ptr`
- (stream path) updates `worker_credits_stream_id` stream register

**What breaks the setter**: If downstream EDM is dead, stalled, or hung (e.g., its own main loop hit a hang, or the ethernet link dropped), credits are never returned.

**Observable symptom**: WAYPOINT stuck at `FWSW`. With fix: periodic `FWST` every ~2-4s.

**Callers**: ~55 call sites across `linear/api.h`, `mesh/api.h`, `tt_fabric_mux.cpp`, `fabric_router_mux_extension.cpp`, `tt_fabric_mux_interface.hpp`, `tt_fabric_udm.hpp`.

**Fix**: Added watchdog counter (100M iterations) that fires `WAYPOINT("FWST")` and resets. Loop continues spinning — caller cannot proceed without a free slot.

**Commit**: `fabric: FIX LT1 — add watchdog to wait_for_empty_write_slot (#42429)`

---

## ITEM 2: Teardown Sync — fabric_erisc_router.cpp

### 2a: `wait_for_other_local_erisc()`

**File**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:2861`

**Exit condition**: Stream scratch register matches expected sync value (`0x0fed` for slave, `0x1bad` for master).

**Who sets it**: The other local ERISC (ERISC0 and ERISC1 on the same ethernet port).

**What breaks the setter**: If one ERISC crashes, hangs in its main loop, or never reaches the teardown sync point, the other spins forever.

**Observable symptom**: No WAYPOINT was emitted (the function had none). With fix: `WAYPOINT("TSYN")` fires after ~2-4s, then the loop breaks out.

**Design note**: The original code had an intentional comment saying "No termination check added — both ERISCs share the same termination signal." This is true for the normal case but fails if one ERISC hangs *before* reaching the sync.

**Fix**: Added bounded iteration count (100M). On timeout, logs `WAYPOINT("TSYN")` and breaks out. The surviving ERISC proceeds with teardown.

**Called 3 times** in `teardown()` (lines 2909, 2935, 2954).

### 2b: `run_drain_step()`

**File**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:2089`

**Status**: Dead code — guarded by `ASSERT(false)` at entry. The while loop already has `got_immediate_termination_signal()` as an exit condition. No fix needed.

**Commit**: `fabric: FIX LT2 — add watchdog to wait_for_other_local_erisc teardown sync (#42429)`

---

## ITEM 3: Dead `wait_for_static_connection_to_ready()` — tt_fabric_mux.cpp

**File**: `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:52`

**Status**: DEAD CODE. Defined but never called from `kernel_main()` or any other function in the translation unit. Unlike the versions in `fabric_router_mux_extension.cpp`, `fabric_router_relay_extension.cpp`, and `fabric_erisc_router.cpp`, this version also lacks a `got_immediate_termination_signal()` check entirely — making it an unbounded spin if it were ever called.

**Fix**: Marked with a deprecation comment for removal.

**Commit**: `fabric: FIX LT3 — mark dead wait_for_static_connection_to_ready in tt_fabric_mux.cpp (#42429)`

---

## ITEM 4: `#ifndef ARCH_WORMHOLE` Termination Guard Removal

**Root cause**: PR #38166 (Sean Nijjar) added `got_immediate_termination_signal()` checks to several init/handshake loops but wrapped them in `#ifndef ARCH_WORMHOLE`, deliberately excluding Wormhole. On WH, these loops have NO escape path if the expected condition never fires.

**Why WH was excluded**: The `got_immediate_termination_signal()` function reads `launch_msg->kernel_config.exit_erisc_kernel` from the mailbox. On WH, this mailbox mechanism may not be reliable during early init, or there may be concerns about read side-effects during the handshake. The exact reason is not documented in the commit.

**On WH, what IS the termination path?**: There is none. If the expected condition (connection request, handshake value, notification value) never arrives, the loop spins forever. The only escape is a board reset or host-side timeout killing the process.

**Fix approach**: Added watchdog counters (100M iterations ~2-4s) to each loop that fire distinct WAYPOINTs. The loops continue spinning — they need the condition to proceed — but hangs are now diagnosable from watcher dumps.

### Affected loops and WAYPOINT codes

```
File                                          Function                          WAYPOINT
─────────────────────────────────────────────────────────────────────────────────────────
tt_fabric_utils.h                             wait_for_notification()           WNTO
fabric_erisc_router.cpp                       wait_for_static_connection...     SCRW
fabric_router_relay_extension.cpp             wait_for_static_connection...     SCRW
fabric_router_mux_extension.cpp               wait_for_static_connection...     SCRW
fabric_router_udm_mux_extension.cpp           wait_for_static_connection...     SCRW
fabric_router_eth_handshake.hpp               fabric_sender_side_handshake()    HSST
fabric_router_eth_handshake.hpp               fabric_receiver_side_handshake()  HSRT
```

**Commit**: `fabric: FIX LT4 — add watchdog logging to WH-unguarded loops (#42429)`

---

## WAYPOINT Code Reference

```
FWSW  — wait_for_empty_write_slot: entered (existing)
FWSD  — wait_for_empty_write_slot: done (existing)
FWST  — wait_for_empty_write_slot: watchdog timeout (NEW)
TSYN  — teardown sync: watchdog timeout, breaking out (NEW)
WNTO  — wait_for_notification: watchdog timeout (NEW)
SCRW  — static connection ready wait: watchdog timeout (NEW)
HSST  — handshake sender: watchdog timeout (NEW)
HSRT  — handshake receiver: watchdog timeout (NEW)
```

## Design Decision: Watchdog-Only vs Break-Out

For most loops, we chose **watchdog-only** (log and continue spinning):
- `wait_for_empty_write_slot()` — cannot proceed without a free slot
- All `wait_for_static_connection_to_ready()` variants — cannot proceed without connection
- Both handshake loops — cannot proceed without peer
- `wait_for_notification()` — cannot proceed without notification

For the teardown sync (`wait_for_other_local_erisc`), we chose **break-out** because:
- The system is already in teardown
- If the other ERISC is dead, waiting forever prevents cleanup
- The subsequent teardown steps can still run (imperfectly) with just one ERISC
