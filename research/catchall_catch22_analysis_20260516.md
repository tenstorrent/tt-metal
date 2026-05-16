<!-- SUMMARY: Deep technical analysis of the ETH handshake catch-22 circular dependencies in fabric init — comparing main vs nsexton/0-racecondition-hunt branch
     KEYWORDS: catch-22, handshake, deadlock, FIX AD, FIX M, FIX S9, relay, MMIO, soft-reset, symmetric, bootstrap, race condition, fabric init
     SOURCE: edm_handshake.hpp, fabric_router_eth_handshake.hpp, fabric_erisc_router.cpp, fabric_init.cpp (both main and racecondition-hunt branch)
     SCOPE: All identified circular dependencies in ETH fabric init — handshake protocol, soft-reset relay bootstrap, prepare_handshake_state timing, MMIO relay paradox
     USE WHEN: Investigating fabric init hangs, STARTED-STARTED deadlocks, relay timeouts on non-MMIO devices, or reviewing the fix strategy for catch-22 class bugs -->

# ETH Handshake Catch-22 Analysis: main vs racecondition-hunt

**Date**: 2026-05-16
**Branch**: `nsexton/0-racecondition-hunt` (FIX AD @ `749979f96f4`)
**Baseline**: `main` (as of 2026-05-16)

---

## Executive Summary

There are **four distinct catch-22 / circular-dependency classes** in the ETH fabric initialization path. Main addresses zero of them. The racecondition-hunt branch eliminates one by construction (Catch-22 #1) and mitigates two others (#3, #4) with defensive workarounds, but Catch-22 #2 (the relay bootstrap paradox) remains fundamentally unsolved and is the current dominant failure mode.

---

## Catch-22 #1: STARTED-STARTED Symmetric Deadlock

### The Problem

On `main`, the ETH handshake uses an asymmetric sender/receiver protocol. One side is designated `is_handshake_sender` (compile-time constant), and the roles differ:

- **Sender** (`fabric_sender_side_handshake`): calls `init_handshake_info()` (zeros `local_value`, preps scratch with MAGIC), then enters a send+poll loop — sends MAGIC to remote's `local_value` every iteration while polling own `local_value` for MAGIC.
- **Receiver** (`fabric_receiver_side_handshake`): calls `init_handshake_info()`, then enters a poll-only loop — waits for MAGIC to appear in own `local_value`, then sends one MAGIC packet back to the sender.

The catch-22: **both peers must agree on who is sender vs receiver**. The `is_handshake_sender` flag is set by the host based on compile-time constants. If both sides end up in receiver role (e.g., due to a firmware restart where one side gets the wrong binary, or an init ordering anomaly), **neither side sends MAGIC** — both sit in poll-only loops forever.

Even in the intended asymmetric case, a subtler deadlock exists: `init_handshake_info()` zeros `local_value` at the START of both `fabric_sender_side_handshake` and `fabric_receiver_side_handshake`. If Side A (sender) has already written MAGIC to Side B's `local_value`, and then Side B enters `fabric_receiver_side_handshake` and calls `init_handshake_info()`, it **erases** Side A's already-delivered MAGIC. Side B then waits for MAGIC that will never come again (sender already exited the loop thinking handshake succeeded). This is the "erase race."

### How Main Addresses It

**It doesn't.** Main relies on the compile-time `is_handshake_sender` assignment being consistent and on timing working out. The erase race exists but is extremely rare because Object Setup takes long enough on both sides that both `init_handshake_info` calls complete before either side sends MAGIC.

### How the Branch Addresses It (FIX AD)

FIX AD eliminates this entire class by construction:

1. **`prepare_handshake_state()`** — a new function called during Object Setup, BEFORE `edm_status = STARTED`. It zeros `local_value` and preps scratch with MAGIC hundreds of microseconds before any handshake loop runs. Since `edm_status` hasn't reached STARTED, no peer can be sending MAGIC yet.

2. **`symmetric_handshake()` / `fabric_symmetric_handshake()`** — replaces the sender/receiver split. Both sides execute the same loop: send MAGIC + poll for MAGIC every iteration. No role assignment needed.

3. A post-loop final send (FIX HS1/HS2) as belt-and-suspenders: if one side exits slightly before the other enters the loop, the departing side sends one more MAGIC to unblock the late-arriving peer.

**Result**: The STARTED-STARTED deadlock class is **CLOSED** by construction. Both the role-mismatch deadlock and the erase race are eliminated.

---

## Catch-22 #2: Relay Bootstrap Paradox (THE DOMINANT OPEN ISSUE)

### The Problem

In a multi-chip system (e.g., T3000 with 8 devices), only devices 0-3 are MMIO-capable (directly accessible via PCIe). Devices 4-7 are non-MMIO — the host reaches them via an ETH relay through one of the MMIO devices. The relay uses base-UMD firmware running on specific ETH channels (typically chan=0 and chan=7 on each device).

The circular dependency:

```
To soft-reset non-MMIO ETH channels:
  → Need to call assert_risc_reset_at_core() on those channels
  → assert_risc_reset_at_core() routes through the ETH relay (UMD WriteToDeviceL1)
  → If the relay channel itself needs resetting, the relay is broken
  → The write never completes → indefinite hang

To keep the relay alive:
  → Must NOT reset the relay channel
  → But the relay channel has dirty state from the prior session
  → Dirty state means fabric firmware handshakes may fail on that channel
```

This is the "relay bootstrap paradox": you need the relay to fix the relay.

### How Main Addresses It

**Main doesn't have any soft-reset logic in `configure_fabric_cores()`** — it simply iterates over channels and writes zeroes to L1 addresses. Main's `configure_fabric_cores()` is 12 lines long:

```cpp
void configure_fabric_cores(tt::tt_metal::IDevice* device) {
    // ... setup ...
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}
```

Main assumes all channels are alive and all L1 writes will succeed. If a channel is dead or the relay is broken, `WriteToDeviceL1` hangs indefinitely. Main simply doesn't encounter this problem as often because it doesn't do soft-resets that could break the relay.

### How the Branch Addresses It (FIX M + FIX S9 + FIX RR)

The branch adds a 370+ line `configure_fabric_cores()` with multiple defensive strategies:

- **FIX M**: Skip soft-reset for base-UMD relay channels on **non-MMIO** devices. Their BRISC serves as the ETH relay endpoint — halting it kills all host communication to that device. `write_launch_msg_to_core` transitions the relay firmware to fabric firmware without a reset, accepting dirty L1/TXQ/MAC state.

- **FIX S9**: On **MMIO** devices, base-UMD relay channels CAN be safely soft-reset because PCIe-direct access has no ETH relay dependency. This cleans the dirty state that FIX M must accept on non-MMIO devices.

- **FIX RR**: For pre-confirmed dead channels on MMIO devices, attempt PCIe-direct soft-reset as a recovery mechanism.

- **FIX TG** (referenced): L1 clear loop is also skipped for non-MMIO base-UMD channels.

**Result**: The paradox is **mitigated but not solved**. Non-MMIO relay channels retain dirty state from the prior session. The AI-JOURNAL (Cycle 24) documents this as the current dominant failure: both relay channels (chan=0 + chan=7) on all 4 non-MMIO devices (4,5,6,7) are in dirty base-UMD state, relay writes timeout at 5s each, all non-MMIO devices excluded from the fabric ring, ring sync master has no peer, and the test fails after a 120-second ring sync timeout.

### True Fix (Not Yet Implemented)

The journal notes "OPTION B (medium term)" — ETH-DMA based non-MMIO channel reset via MMIO fabric firmware. This would break the catch-22 by using an already-initialized MMIO device's fabric firmware to send reset commands to non-MMIO channels over the ETH DMA path, bypassing the UMD relay entirely.

---

## Catch-22 #3: prepare_handshake_state Timing Window

### The Problem

Even with FIX AD's `prepare_handshake_state()` being called during Object Setup, there is a residual timing question: what if Object Setup timing overlaps between the two peers?

The sequence on each side is:
1. `edm_status = STARTED` (line 3251 in branch)
2. `prepare_handshake_state()` — zeros `local_value` (line 3260 in branch)
3. ... Object Setup continues (hundreds of lines of channel buffer allocation) ...
4. `wait_for_other_local_erisc()` barrier (line 3582 in branch)
5. `fabric_symmetric_handshake()` — starts sending MAGIC (line 3590 in branch)

The concern: Side A reaches step 5 and sends MAGIC to Side B's `local_value`. Side B is still at step 2 and zeros `local_value`, erasing Side A's MAGIC. This would recreate the original erase race.

### Why It's Actually Safe (In the Branch)

This timing window does NOT manifest because of two ordering guarantees:

1. **`edm_status = STARTED` is set BEFORE `prepare_handshake_state()`** (line 3251 before line 3260). The host uses `edm_status == STARTED` as the gate to confirm firmware is running. The peer's host won't have triggered firmware launch until its own side also reaches STARTED.

2. **Object Setup takes hundreds of microseconds** (channel buffer allocation, sender/receiver interface construction). Both sides must complete this AND the `wait_for_other_local_erisc()` barrier before entering the handshake loop. The `prepare_handshake_state()` call at step 2 is thus hundreds of microseconds BEFORE step 5's send.

3. **Both devices launch firmware nearly simultaneously** (the host launches all devices in a tight loop). Even in the worst case, the skew between the two sides completing Object Setup is much smaller than the Object Setup duration itself.

However, the safety argument depends on timing margins, not a hard ordering guarantee. If Object Setup were ever shortened or the launch sequence changed, this window could reopen.

### How Main Is Affected

Main has the same conceptual window but worse: `init_handshake_info()` zeros `local_value` AT THE START of the handshake function itself (after Object Setup). If one side is slow to reach the handshake point, the other side could already be sending. Main relies on the same timing margins.

---

## Catch-22 #4: FIX M / MMIO Relay Clean-State Paradox

### The Problem

FIX M skips soft-reset for non-MMIO relay channels to keep the relay alive. But skipping soft-reset means those channels retain dirty L1, dirty ETH TXQ state, and dirty MAC state from the prior session. This dirty state can cause:

- **Stale `edm_status`** values (STARTED = 0xa0b0c0d0, REMOTE_HANDSHAKE_COMPLETE = 0xa1b1c1d1) persisting from the prior session, confusing health checks
- **Stale ETH TXQ commands** — if the prior firmware was mid-DMA-transfer when halted, `ETH_TXQ_CMD` is non-zero, causing the next `eth_send_packet()` to spin forever on `eth_txq_is_busy()`
- **Stale L1 handshake values** — `local_value` might contain MAGIC from the prior session

FIX AD's `prepare_handshake_state()` partially addresses the stale TXQ and stale L1 problems:

- **FIX AH** (inside `prepare_handshake_state`): Flushes stale ETH TXQ state by writing `ETH_TXQ_CMD_FLUSH` when `eth_txq_is_busy()` is true. But with a critical guard: on Wormhole, writing FLUSH to an already-idle queue may not auto-clear, causing a second hang.
- The `local_value = 0` write cleans the stale handshake value.

But the dirty `edm_status` value is NOT cleaned by firmware-side code — it's supposed to be cleared by the host's L1 clear loop in `configure_fabric_cores()`. For non-MMIO channels, FIX TG skips the L1 clear (because WriteToDeviceL1 routes through the potentially-broken relay), so `edm_status` retains its stale value.

### How Main Is Affected

Main doesn't do soft-resets at all, so it never intentionally leaves dirty state. But it also never cleans dirty state from process kills / SIGKILL scenarios. The L1 clear loop in main's `configure_fabric_cores()` runs on all channels including non-MMIO ones, which works ONLY if the relay is healthy. If the relay is broken (prior session crashed), main hangs indefinitely in the L1 clear loop.

### Branch Mitigation

The branch accepts dirty state on non-MMIO relay channels as the lesser evil compared to hanging. It relies on:
- FIX AH to flush stale TXQ (firmware-side)
- FIX AD to zero stale `local_value` (firmware-side)
- The firmware's ability to boot into fabric mode via `write_launch_msg_to_core` even with dirty L1

The unresolved gap: if the relay itself is broken (non-MMIO relay channels unresponsive), there is no mechanism to clean them. The branch detects this condition (FIX NX timeout) and excludes the affected non-MMIO devices from the fabric, but this degrades functionality.

---

## Summary Comparison

```
Catch-22                            main           Branch (racecondition-hunt)
────────────────────────────────────────────────────────────────────────────────
#1 STARTED-STARTED deadlock         Vulnerable     CLOSED (FIX AD symmetric)
#2 Relay bootstrap paradox          Not addressed  Mitigated (FIX M/S9/RR/TG)
                                    (hangs)        (degrades non-MMIO instead)
#3 prepare_handshake_state timing   Vulnerable     Safe by timing margin
                                    (same class)   (not hard guarantee)
#4 MMIO relay clean-state paradox   Not addressed  Partially mitigated (FIX AH)
                                    (hangs)        (stale edm_status unsolved)
```

## Remaining Gaps

1. **Catch-22 #2 is the dominant open failure.** When all non-MMIO relay channels are dirty, all 4 non-MMIO devices are excluded. The ring sync master has no peer and times out at 120 seconds. Proposed fix: ETH-DMA based reset via MMIO firmware (OPTION B).

2. **Catch-22 #3's safety is timing-dependent**, not formally guaranteed. A sufficiently fast Object Setup or asymmetric firmware launch could reopen the window. Consider adding an L1 flag or barrier to make the ordering explicit.

3. **Catch-22 #4's stale `edm_status` on non-MMIO channels** confuses downstream health checks. The firmware could zero its own `edm_status` address as part of `prepare_handshake_state()`, but this would need to happen before the host reads it for health monitoring.

4. **FIX DW** (50ms delay before FIX DU heartbeat poll) is identified as needed but not yet confirmed landed — prevents false-positive "0ms" recovery reports on MMIO channels.

---

## File References

- **Branch `edm_handshake.hpp`**: `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp` — `prepare_handshake_state()`, `symmetric_handshake()`, FIX AH, diagnostic fields
- **Branch `fabric_router_eth_handshake.hpp`**: `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp` — `fabric_symmetric_handshake()` with termination signal support
- **Branch `fabric_erisc_router.cpp`**: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` — lines 3254-3264 (prepare_handshake_state callsite), lines 3585-3594 (symmetric_handshake callsite)
- **Branch `fabric_init.cpp`**: `tt_metal/fabric/fabric_init.cpp` — lines 93-470+ (FIX M, FIX S9, FIX RR, FIX BH, FIX DU, dead channel tracking)
- **Main `edm_handshake.hpp`**: Same path — `sender_side_handshake()`, `receiver_side_handshake()` (asymmetric)
- **Main `fabric_router_eth_handshake.hpp`**: Same path — `fabric_sender_side_handshake()`, `fabric_receiver_side_handshake()` (asymmetric with termination)
- **Main `fabric_init.cpp`**: Same path — 12-line configure_fabric_cores (no soft-reset, no dead channel tracking)
