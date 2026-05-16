<!--
SUMMARY: Investigation of whether staged ERISC initialization (MMIO-first, non-MMIO-second) can eliminate the simultaneous-handshake deadlock in fabric firmware init
KEYWORDS: staged-init, ETH, handshake, deadlock, ERISC, MMIO, non-MMIO, FIX-AE, FIX-BE, FIX-AF, local-handshake, EDMStatus, STARTED, REMOTE_HANDSHAKE_COMPLETE, LOCAL_HANDSHAKE_COMPLETE, READY_FOR_TRAFFIC, T3K, quiesce, configure_fabric, racecondition-hunt
SOURCE: Code analysis of nsexton/0-racecondition-hunt branch (tt_metal codebase)
SCOPE: Feasibility analysis of staged initialization, EDM state machine walkthrough, evaluation of 4 deadlock hypotheses, gap analysis of init-path vs quiesce-path
USE WHEN: Investigating ETH handshake deadlocks, evaluating ERISC init ordering strategies, understanding why channels get stuck at REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1)
-->

# Staged ERISC Initialization: Feasibility Investigation

**Date**: 2026-05-15
**Branch**: `nsexton/0-racecondition-hunt`
**Author**: BrAIn (AI assistant investigation, requested by Neil)

---

## 1. EDM State Machine

The ERISC Data Mover (EDM) firmware progresses through four states, defined as `EDMStatus` values:

```
STARTED (0x1)
  → written at fabric_erisc_router.cpp:3339, BEFORE object setup
  → ~300 lines of Object Setup code execute after this
  → then wait_for_other_local_erisc() barrier
  → then ETH handshake (sender or receiver)

REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1)
  → written at fabric_erisc_router.cpp:3688, after ETH handshake succeeds
  → ETH peer link is alive

LOCAL_HANDSHAKE_COMPLETE
  → written at fabric_erisc_router.cpp:3706
  → ALL ERISCs on the same device have completed their ETH handshakes
  → Master ERISC waited for subordinates, then notified them

READY_FOR_TRAFFIC
  → written by HOST (not firmware) after Phase 5
  → Master ERISC propagates to subordinates via NOC
```

### Key asymmetry in the ETH handshake

The handshake protocol (`fabric_router_eth_handshake.hpp`) assigns roles at compile time:

- **Sender** (lower chip_id, typically MMIO): Loops sending `MAGIC_HANDSHAKE_VALUE` to peer via `eth_send_packet`, waits for local value to be overwritten with MAGIC by peer's ack.
- **Receiver** (higher chip_id, typically non-MMIO): Waits for local value to equal MAGIC (sent by peer), then sends ack back.

On Wormhole: the termination signal check is compiled out. If the peer never responds, the handshake loop spins forever. A watchdog at 100M iterations logs a WAYPOINT but does NOT break out.

### The local handshake barrier

After ETH handshake, a critical intra-device synchronization occurs:

```cpp
// fabric_erisc_router.cpp:3691-3704
if constexpr (is_local_handshake_master) {
    // Wait for ALL subordinate ERISCs on this device to reach REMOTE_HANDSHAKE_COMPLETE
    wait_for_notification(edm_local_sync_ptr, num_local_edms - 1, ...);
    // Then notify all subordinates to advance
    notify_subordinate_routers(edm_channels_mask, ...);
} else {
    // Tell master "I'm done with ETH handshake"
    notify_master_router(local_handshake_master_eth_chan, ...);
    // Wait for master to say "everyone is done"
    wait_for_notification(edm_local_sync_ptr, num_local_edms, ...);
}
```

This creates a **device-wide barrier**: the master ERISC on Device 0 cannot advance past `REMOTE_HANDSHAKE_COMPLETE` until every subordinate ERISC on Device 0 also completes its ETH handshake. If Device 0 has ERISCs connecting to Devices 4, 5, 6, and 7 (non-MMIO), those non-MMIO peers must be running for Device 0's ERISCs to complete.

---

## 2. What FIX AE Already Does (and Why It's Not Enough)

### The three-pass quiesce launch (mesh_device.cpp:1670-1793)

FIX AE already implements staged initialization in `quiesce_internal()`:

```
Pass 1a: ALL devices — setup everything, defer ETH launch
Pass 1b: MMIO devices — launch ETH, wait for STARTED (one at a time, per FIX BE)
Pass 1c: Non-MMIO devices — launch ETH, wait for STARTED (one at a time, per FIX AF)
Pass 2:  ALL devices — wait_for_fabric_workers_ready()
```

### What "wait for STARTED" actually means

`wait_for_eth_cores_launched()` (device.cpp:2611) polls each ETH channel for a non-zero `edm_status`. Since `STARTED` is written at line 3339 — **before** Object Setup and **before** the ETH handshake — this checkpoint only guarantees:

> "The ERISC has begun executing firmware and set its status word to 1."

It does NOT guarantee:
- Object Setup is complete (~300 lines of initialization code)
- The ERISC has entered the handshake loop
- The ERISC is ready to respond to its peer's handshake messages

### Why STARTED is the best available checkpoint

The next status written is `REMOTE_HANDSHAKE_COMPLETE` (0xa1b1c1d1), but waiting for this would create a circular dependency:

1. Host launches MMIO Device 0 ETH
2. Host waits for Device 0 to reach REMOTE_HANDSHAKE_COMPLETE
3. Device 0's ERISCs need non-MMIO peers (Devices 4-7) to be running for their handshakes to complete
4. But non-MMIO devices haven't been launched yet (waiting for step 2)
5. **Deadlock**: host waits for Device 0, Device 0 waits for Devices 4-7, Devices 4-7 wait for host

The local handshake compounds this: even if one ERISC on Device 0 connects to an MMIO peer and completes, the master ERISC still waits for ALL subordinate ERISCs, some of which connect to not-yet-launched non-MMIO devices.

Therefore `STARTED` is the highest checkpoint the host can safely wait for without introducing a host-side deadlock.

### The timing gap assumption

FIX AE relies on a timing gap between STARTED and the ETH handshake loop:

```
STARTED → ~300 lines Object Setup → wait_for_other_local_erisc() → ETH handshake
```

The assumption: by the time Device N+1's ERISCs start executing, Device N's ERISCs have progressed through Object Setup and are in (or past) the handshake loop. FIX AF strengthened this by explicitly waiting for STARTED before launching the next device.

But this is a **timing assumption**, not a protocol guarantee. If Object Setup completes faster than the relay launch latency (possible when relay is <1ms in base_umd mode), both sides can still enter the handshake loop simultaneously.

---

## 3. The Init-Path Gap

**Critical finding**: `configure_fabric()` (device.cpp:411-700+) does NOT have three-pass staging. It launches all ETH cores in a single pass within a per-device loop. There is no MMIO-first / non-MMIO-second ordering, no inter-device STARTED barrier, no FIX AE protections.

The three-pass staging only exists in `quiesce_internal()`. On initial session startup, the init-path calls `configure_fabric()` per-device without cross-device coordination.

This means the init-path is **more vulnerable** to the simultaneous-handshake deadlock than the quiesce path.

---

## 4. Hypothesis Evaluation

### Hypothesis A: "Both sides enter the handshake loop simultaneously"
**Verdict: POSSIBLE but mitigated by FIX AE**

The asymmetric sender/receiver protocol means simultaneous entry is only a problem if both sides are senders (which shouldn't happen — role is determined by chip_id). However, FIX HS2 (the post-loop final send) was added specifically to handle a race where both sides finish their handshake within a narrow window. The scenario where both senders loop simultaneously is prevented by the compile-time role assignment.

In the quiesce path with FIX AE, the STARTED barrier reduces (but doesn't eliminate) the probability of simultaneous entry.

### Hypothesis B: "Stale ETH state from prior crash"
**Verdict: PLAUSIBLE and not fully addressed**

After a prior crash, L1 memory may retain stale values:
- `MAGIC_HANDSHAKE_VALUE` in the handshake locations could cause a receiver to immediately see "magic" and ack a non-existent sender
- Old EDMStatus values could confuse the host-side polling

The canary mechanism (0xDEADB07E host pre-launch canary, 0xA0A0A0A0 firmware canary) partially addresses this for the status word, but the handshake L1 locations are separate from the status word. `configure_fabric_cores()` clears L1, but the clear may be partial or skipped for channels in `skip_soft_reset_channels`.

### Hypothesis C: "Non-MMIO device takes too long to relay-launch"
**Verdict: ADDRESSED by FIX AF**

FIX AE originally assumed ~200ms relay latency for the gap, but relay was <1ms when Device 0 was in base_umd mode. FIX AF fixed this by explicitly polling for STARTED before launching the next device. This hypothesis was the original root cause of the FIX AE failure.

### Hypothesis D: "Local handshake deadlock — channels stuck at REMOTE_HANDSHAKE_COMPLETE"
**Verdict: MOST LIKELY residual failure mode**

This is the `bc_deadlock` pattern detected by Phase 5b health check. The scenario:

1. Device 0 has 4 ERISCs: chans A, B, C, D connecting to Devices 4, 5, 6, 7 respectively
2. Chan A (Device 0 → Device 4) completes ETH handshake → REMOTE_HANDSHAKE_COMPLETE
3. Chan B (Device 0 → Device 5) is still in handshake (Device 5 not yet launched)
4. Master ERISC on Device 0 waits for ALL subordinates at the local handshake barrier
5. Chan A is stuck at REMOTE_HANDSHAKE_COMPLETE indefinitely, waiting for Chan B
6. If Chan B's peer (Device 5) also has channels waiting for Device 0 to advance → deadlock

The local handshake barrier creates a **cross-device dependency chain**: Device 0 cannot advance until all its peers respond, but some peers' devices cannot advance until Device 0 responds to their other channels. In a fully connected T3K mesh, this creates potential for circular waits.

FIX AE's sequential launch order mitigates this by ensuring all senders are running before receivers start, but the local handshake barrier means "running" isn't enough — the sender must actually complete its ETH handshake before the master can advance, and that requires the receiver (on a device launched later) to also be running.

---

## 5. Can Staged Init Fully Solve the Deadlock?

### Short answer: No — not with the current protocol

Staged initialization (MMIO-first, non-MMIO-second) cannot fully eliminate the deadlock because:

1. **The local handshake creates circular dependencies** that no launch ordering can break. MMIO Device 0's master ERISC needs ALL its subordinate ERISCs to complete their ETH handshakes. Some connect to non-MMIO peers. Those non-MMIO peers can't be launched until MMIO is "done." But MMIO can't be "done" until non-MMIO peers respond.

2. **STARTED is too early** — it's written before the ~300-line Object Setup phase, so waiting for STARTED doesn't guarantee the ERISC is ready to handshake. It only reduces the probability of simultaneous entry.

3. **There is no safe intermediate checkpoint** between STARTED and REMOTE_HANDSHAKE_COMPLETE that means "ready to receive handshake messages."

### What would actually fix it

To truly eliminate the deadlock, the protocol itself needs changes:

**Option 1: Remove the local handshake barrier** — Let each ERISC advance independently after its own ETH handshake completes. The local barrier exists for synchronization guarantees (all channels ready before traffic), but if these guarantees can be provided by the host-side READY_FOR_TRAFFIC gate instead, the barrier is redundant.

**Option 2: Add a "ready to handshake" status** — Insert a new EDMStatus between STARTED and REMOTE_HANDSHAKE_COMPLETE (e.g., `HANDSHAKE_READY = 0x2`) written after Object Setup completes and the ERISC enters the handshake loop. The host waits for this before launching the peer. This is stronger than STARTED because it guarantees the ERISC is actually in the handshake loop.

**Option 3: Add timeout + retry to the handshake** — Instead of spinning forever, add a timeout to the handshake loop. If the peer doesn't respond within N iterations, reset the handshake state and retry. This converts the deadlock into a recoverable condition. (On WH, the termination signal check is compiled out, so this would need a different mechanism.)

**Option 4: Session nonce** — Include a per-session nonce in the handshake magic value. This prevents stale L1 values from prior sessions from being misinterpreted as valid handshake messages. See `research/eth_handshake_deadlock_strategies.md` for detailed design.

---

## 6. Summary

```
Question                                    Answer
─────────────────────────────────────────── ──────────────────────────────────────
Q1: Can staged init eliminate the deadlock?  No — local handshake barrier creates
                                             circular deps no ordering can break.

Q2: What does "fully initialized" mean?      STARTED = firmware started (too early).
                                             REMOTE_HANDSHAKE_COMPLETE = ETH done
                                             (needs peer, creates circular dep).
                                             No safe intermediate checkpoint exists.

Q3: Does FIX AE already do this?             Yes — three-pass launch in quiesce_internal.
                                             But only waits for STARTED, not handshake.

Q4: Init-path has same protection?           No — configure_fabric() has NO staging.
                                             Single-pass, no cross-device barriers.

Q5: What hypothesis is most likely?          D — local handshake barrier deadlock.
                                             Channels reach REMOTE_HANDSHAKE_COMPLETE
                                             but master waits for ALL subordinates,
                                             creating cross-device circular waits.
```

**Recommendation**: Staged init is necessary but insufficient. The real fix requires protocol-level changes (options 1-4 above). The init-path (`configure_fabric()`) should get FIX AE staging as a minimum defense-in-depth measure, but this alone won't eliminate the deadlock.
