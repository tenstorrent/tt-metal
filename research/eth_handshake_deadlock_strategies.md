<!--
SUMMARY: Design analysis of ETH handshake deadlock strategies for T3K AllGather hangs — root cause, FIX HS1/HS2 assessment, and evaluation of 7 mitigation strategies with implementation plans
KEYWORDS: ETH, handshake, deadlock, AllGather, T3K, ERISC, fabric, Wormhole, sender, receiver, MAGIC_HANDSHAKE_VALUE, FIX-HS1, FIX-HS2, race-condition, init_handshake_info, edm_handshake
SOURCE: Code analysis of nsexton/0-racecondition-hunt branch — edm_handshake.hpp, fabric_router_eth_handshake.hpp, fabric_erisc_router.cpp, erisc_datamover_builder.cpp, fabric_firmware_initializer.cpp
SCOPE: Handshake protocol race conditions, role assignment mechanism, ARCH_WORMHOLE termination gap, L1 state across resets, all 7 proposed mitigation strategies (A–G)
USE WHEN: Investigating simultaneous-sender handshake deadlocks, designing handshake protocol improvements, evaluating defense-in-depth for fabric init reliability
-->

# ETH Handshake Deadlock: Root Cause Analysis and Mitigation Strategies

## 1. Root Cause Analysis

### 1.1 The Handshake Protocol

The ERISC-to-ERISC handshake synchronizes two ethernet-connected cores before traffic flows.  The protocol lives in two layers:

- **Base layer** (`edm_handshake.hpp`): `sender_side_handshake()` and `receiver_side_handshake()`
- **Fabric layer** (`fabric_router_eth_handshake.hpp`): `fabric_sender_side_handshake()` and `fabric_receiver_side_handshake()` — identical logic plus termination-signal support (which is `#ifdef`'d out on Wormhole)

The data structure (`handshake_info_t`, lines 49–56 of `edm_handshake.hpp`):

```
struct handshake_info_t {
    uint32_t local_value;        // Written by remote with 0xAA
    uint16_t neighbor_mesh_id;   // Populated via DMA from scratch[1]
    uint8_t  neighbor_device_id;
    uint8_t  padding0;
    uint32_t padding[2];
    uint32_t scratch[4];         // scratch[0]=MAGIC, scratch[1]=identity
};
```

**`init_handshake_info()`** (lines 58–88):
1. Flushes stale ETH TX queue (FIX AH) — only if `eth_txq_is_busy()`
2. Sets `local_value = 0`
3. Sets `scratch[0] = 0xAA` (MAGIC_HANDSHAKE_VALUE)
4. Sets `scratch[1] = mesh_id | (device_id << 16)`

**Sender loop** (lines 90–137): repeatedly calls `eth_send_packet(0, scratch_addr, local_val_addr, 1)` — this DMA-copies the local `scratch[0..3]` (16 bytes) to the remote's `local_value` field (bytes 0–15). Exits when `local_value == MAGIC`. After exit, sends one final unconditional packet (FIX HS1).

**Receiver loop** (lines 139–167): polls `local_value` until it equals MAGIC. Then sends one packet back (scratch → remote's local_value).

### 1.2 Role Assignment

Role is determined at **compile time** in `erisc_datamover_builder.cpp` (lines 884–893):

```cpp
bool is_handshake_master = local_tie_break_id < peer_tie_break_id;
```

Where:
- Same mesh: compare `chip_id` values (lower wins)
- Different meshes: compare `mesh_id` values (lower wins)

This is emitted as the compile-time arg `IS_HANDSHAKE_SENDER` (line 1174), consumed in `fabric_erisc_router_ct_args.hpp` (line 129):

```cpp
constexpr bool is_handshake_sender = NAMED_CT_ARG("IS_HANDSHAKE_SENDER") != 0;
```

And dispatched at `fabric_erisc_router.cpp` lines 3660–3675:

```cpp
if constexpr (enable_ethernet_handshake) {
    if constexpr (is_handshake_sender) {
        fabric_sender_side_handshake<...>(...);
    } else {
        fabric_receiver_side_handshake<...>(...);
    }
}
```

**Critical finding: roles are asymmetric by design.** For any given ETH link, one side is always compiled as sender and the other as receiver. The tiebreak is deterministic (lower chip_id wins). Both sides CANNOT be compiled as sender for the same link.

This invalidates the premise that "both sides call sender_side_handshake() concurrently" in the normal case. However, FIX HS1/HS2 still addresses a real race — see Section 2.

### 1.3 The Actual Race Window (FIX HS1/HS2 Scenario)

Even with asymmetric roles, the sender-before-receiver race is real:

1. Side A (sender, MMIO device, chip_id=0) starts first via fast relay (~1ms)
2. Side A enters `sender_side_handshake()`, calls `init_handshake_info()`, starts sending MAGIC to B's `local_value`
3. Side A's sends arrive at B's L1 — writing 0xAA to B's `handshake_info_t.local_value`
4. Side B (receiver, non-MMIO device, chip_id=1) starts ~200ms later via slow relay
5. **B's `init_handshake_info()` sets `local_value = 0`** — erasing A's 0xAA
6. B enters `receiver_side_handshake()` — polls `local_value`, which is now 0
7. Meanwhile, B's receiver loop sends nothing (receiver doesn't send until it sees MAGIC)
8. A already exited its loop (it saw MAGIC from... wait — A is sender, B is receiver. B hasn't sent anything to A yet)

Wait. Let's re-trace with correct roles:
- A = sender: loops sending MAGIC to B, waiting for B to write MAGIC to A's `local_value`
- B = receiver: polls `local_value` for MAGIC, then sends one reply

**Race scenario with correct roles:**
1. A starts, enters sender loop, sends MAGIC → B.local_value repeatedly
2. Some sends land before B starts. B.local_value = 0xAA
3. B starts. `init_handshake_info()` clears B.local_value = 0. **All of A's prior sends are erased.**
4. B enters receiver loop. Polls B.local_value (which is 0)
5. A is still in sender loop — **A is still sending MAGIC → B.local_value** on every iteration
6. A's next send after B's init arrives at B. B.local_value = 0xAA. B sees MAGIC, sends reply, exits
7. A receives B's reply, sees MAGIC, exits

This actually works without FIX HS1. As long as A continues sending while B is running, B will eventually see MAGIC. The race is only possible if:

**The dangerous race (FIX HS1 addresses):**  When both sides happen to call `sender_side_handshake()` — which contradicts the compile-time assignment. OR: when the compile-time args are somehow swapped/corrupted. OR: in the **deprecated** handshake path (non-fabric, legacy CCL).

But there's a subtler issue FIX HS1 addresses even with correct roles:

**ETH TX queue timing:** `eth_send_packet()` waits for `eth_txq_is_busy()` first (line 85 of tunneling.h). If the TXQ is busy from a prior iteration, A may spin on TXQ-busy for microseconds. During that spin, B's `init_handshake_info()` could clear and re-read. However, this only delays A's next send — it doesn't stop A from eventually sending. So this is not a real deadlock.

**The real value of FIX HS1** is defensive: it guarantees the protocol is correct even if roles are ever accidentally made symmetric (e.g., compile-time arg bug, link with equal chip_ids, etc.). It costs one extra `eth_send_packet` call — essentially free.

### 1.4 The `ARCH_WORMHOLE` Termination Gap

In `fabric_router_eth_handshake.hpp`, the while-loop condition is:

```cpp
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
       && !tt::tt_fabric::got_immediate_termination_signal<...>(termination_signal_ptr)
#endif
)
```

On Wormhole, `got_immediate_termination_signal()` is compiled out. This means:
- If the peer ERISC never starts (hardware failure, broken link, dead relay), this loop spins **forever**
- The host's only recourse is `assert_risc_reset_at_core` (RISC hard reset via CSR write) — this halts the RISC but does NOT reset the ETH MAC/DMA
- The watchdog WAYPOINT ("HSST" / "HSRT") fires every 100M iterations for diagnostics, but provides no escape

`got_immediate_termination_signal()` (in `tt_fabric_utils.h`, lines 24–31) checks two things:
1. `termination_signal_ptr == IMMEDIATELY_TERMINATE` — host writes to L1
2. `launch_msg->kernel_config.exit_erisc_kernel` — mailbox-based exit

On Wormhole, neither is checked during handshake. The host must perform a hard RISC reset.

### 1.5 L1 State After Soft Reset

From the FIX AH comments in `init_handshake_info()` (lines 60–76):

> "ERISC soft-reset halts the RISCV core but does NOT reset ETH MAC/DMA hardware. If the prior firmware was terminated while an ETH TX was in-flight, ETH_TXQ_CMD remains non-zero."

Key facts:
- **L1 memory survives RISC soft-reset** — only the RISC-V core state (PC, registers) is reset
- `handshake_info_t.local_value` at its reserved L1 address retains whatever was written by the prior firmware instance
- ETH MAC/DMA state also survives — pending TX operations remain in the queue
- FIX AH flushes the stale TX queue, but only the local TX side. **There is no mechanism to drain pending RX** — a packet in-flight from the remote that lands after `init_handshake_info()` sets `local_value=0` could write 0xAA to local_value, making the ERISC think its peer has already started when in fact the 0xAA is from the prior session

### 1.6 The MAGIC_HANDSHAKE_VALUE (0xAA) Weakness

`MAGIC_HANDSHAKE_VALUE = 0xAA` is an 8-bit sentinel in a 32-bit field. After a crash:
- L1 may contain 0xAA from the prior session at the `local_value` address
- `init_handshake_info()` resets it to 0, but a stale in-flight ETH RX DMA could overwrite it back to 0xAA after the reset
- The 0xAA value has no session-id, nonce, or sequence number — there is no way to distinguish "fresh handshake from current peer" vs "stale leftover from crashed session"

This is a real gap. In practice it is mitigated by timing: the stale in-flight RX would need to land in the narrow window between `init_handshake_info()` clearing `local_value=0` and the new handshake loop starting. At ~1us ETH latency, this window is extremely narrow but non-zero.


## 2. Assessment of FIX HS1/HS2

### 2.1 What They Fix

FIX HS1 (`edm_handshake.hpp`, line 136) and FIX HS2 (`fabric_router_eth_handshake.hpp`, lines 66/69) add one unconditional `eth_send_packet` after the sender's while-loop exits.

This handles:
- **Same-role race (defensive):** If both sides are somehow in sender mode, the first to exit sends one final MAGIC to unblock the other
- **Late-arriving sender:** If sender A exits before B's last `init_handshake_info()` erases its writes, the final post-loop send arrives after B's reset and unblocks B

### 2.2 Are They Sufficient?

**For the designed protocol (asymmetric roles): YES.** With compile-time role assignment:
- Sender continuously sends MAGIC to receiver's `local_value`
- Receiver waits for MAGIC, then replies
- Even if receiver's `init_handshake_info()` erases the first few sends, sender keeps sending until it sees the reply
- FIX HS1's final send is unnecessary but harmless for asymmetric roles

**For the hypothetical symmetric-sender case: MOSTLY.** FIX HS1 addresses the primary race but has a theoretical gap:

**Remaining edge case:** If both sides start truly simultaneously (both call `init_handshake_info()` at the same time), both set `local_value=0`, both enter their sender loops, both start sending MAGIC. Both will see MAGIC from the other, both will exit, both will send the final packet. This actually works correctly — FIX HS1 is not even needed here.

The race FIX HS1 specifically addresses is the **staggered start** case: A starts early, B starts late, A exits before B starts, B's init erases A's sends, and without FIX HS1, nobody would send to B again. With FIX HS1, A's final send arrives after B's init, unblocking B.

**Verdict: FIX HS1/HS2 are correct and sufficient for all known scenarios.** The one theoretical gap (stale RX DMA from prior session) is not addressed by FIX HS1/HS2 but is a different class of bug.

### 2.3 Conditions Where Deadlock Can Still Occur

1. **Wormhole termination gap:** If peer never starts (hardware failure), sender spins forever. Not a handshake-protocol bug — it's a missing escape hatch.
2. **Stale RX from prior session:** A crashed session's in-flight ETH packet could write 0xAA to `local_value` after `init_handshake_info()` resets it, causing premature handshake completion with a peer that isn't actually running. This would cause the router to proceed to the main loop and hang when the peer never sends real traffic.
3. **ETH TXQ busy-loop after reset:** If `eth_txq_is_busy()` returns true indefinitely (MAC hardware wedged), `eth_send_packet()` inside the sender loop spins forever. FIX AH handles this for `init_handshake_info()` but not for subsequent sends in the loop body.


## 3. Strategy Evaluation

### Strategy A: Asymmetric Role Assignment via Stable Chip Property

**Description:** Use MMIO vs non-MMIO (or physical chip_id) to deterministically assign sender/receiver at runtime.

**Assessment:** This is already the current design. The role is assigned at compile time via `is_handshake_master = local_tie_break_id < peer_tie_break_id` in `erisc_datamover_builder.cpp` (line 893). The lower chip_id (or mesh_id for cross-mesh links) always gets sender. This IS a stable chip property — chip_ids are derived from physical connectivity and do not change at runtime.

**Verdict: Already implemented. No action needed.** The assertion `local_fabric_node_id != peer_fabric_node_id` (line 882) guarantees the tiebreak always produces a winner.

**Edge case:** If two chips somehow have the same fabric_node_id (a control plane bug), `is_handshake_master` would always be false on both sides → both would be receiver → deadlock. The `TT_ASSERT` on line 882 guards against this.

**Complexity:** N/A (done)
**Wormhole concern:** None — role assignment is arch-independent


### Strategy B: Collision Detection with Backoff (CSMA-CD)

**Description:** Both sides start as tentative senders. Write a canary (device_id) to a "who-is-sending" L1 field. After a spin, read back — if other side's ID is there, switch to receiver. Lower device_id wins.

**Assessment:**
- **Does not address root cause:** The root cause is not symmetric roles (Strategy A already handles that). This adds complexity without solving a real problem.
- **Race window:** Two `eth_send_packet` calls are not atomic. If both sides write within the same ETH roundtrip (~1us on Wormhole), both will see their own value (because the remote's write is still in-flight). Need at least 2 roundtrips to resolve.
- **No read-back primitive:** `eth_send_packet` only writes TO the remote. There is no direct read-from-remote via ETH DMA. The only way to "read" the remote is to have the remote send its value to you — which is what the existing handshake already does.

**Verdict: Rejected. Not implementable with current ETH primitives (no remote read), and solves a problem that doesn't exist (roles are already asymmetric).**

**Complexity:** High — requires new L1 protocol state, collision detection logic, backoff timing
**Wormhole concern:** No remote-read capability over ETH; would need to be simulated via send+reply


### Strategy C: Rendezvous Barrier with Sequence Numbers

**Description:** Replace send-until-ack with sequence-number barrier. Both sides write monotonically incrementing sequence numbers. Higher sequence number is initiator. Equal → device_id tiebreak.

**Assessment:**
- **Addresses stale-state problem:** Session-specific sequence numbers can distinguish "current session" from "stale leftover." A sequence number that survives reset would prevent false handshake completion from prior-session data.
- **L1 persistence:** L1 survives soft reset (confirmed by FIX AH comments). A sequence number stored in a reserved L1 word would persist and be readable by the next firmware instance.
- **Implementation:** Reserve 4 bytes of L1 for `handshake_sequence_number`. At startup, read it, increment by 1, write it to scratch, send to peer. Peer compares received sequence number against expected (its own + 1 or similar scheme).
- **Challenge:** Both sides must agree on what constitutes a "fresh" sequence number. If A resets but B doesn't, A's sequence number goes up by 1 while B still has the old one. The protocol must handle asymmetric resets.

**Verdict: Promising for addressing the stale-RX-after-reset gap (Section 1.6). Not needed for the primary race (already fixed). Should be considered as defense-in-depth if stale-RX false completions are observed in practice.**

**Complexity:** Medium — firmware: 1 new L1 word, small logic change in init; host: reserve L1 address, no clear/init of that word
**Wormhole concern:** None — sequence numbers are arch-independent


### Strategy D: Host-Mediated Role Assignment

**Description:** Host writes a role word to each ERISC's L1 before firmware launch. Firmware reads role at startup and branches.

**Assessment:**
- **Redundant with current design:** Role is already embedded in compile-time args (`IS_HANDSHAKE_SENDER`). Compile-time args are written to L1 by the kernel loader before ERISC deassert. This IS host-mediated role assignment, just via compile-time args instead of a separate L1 word.
- **Adds no value over current approach:** The compile-time arg is set by the same host-side code that would write the L1 word. Both approaches have the same timing properties — the write must complete before ERISC firmware reads it.
- **Relay race concern mentioned in prompt:** "relay write for non-MMIO side may not arrive before ERISC comes up" — this is the same concern that applies to compile-time arg delivery. In practice, the host launches ERISC after writing CT args (deassert comes after write), so this is not a real race.

**Verdict: Rejected — functionally identical to current compile-time arg approach with extra complexity.**

**Complexity:** Low, but pointless
**Wormhole concern:** Same as current approach — none


### Strategy E: Timeout-and-Yield in Firmware

**Description:** Sender enters a bounded iteration count. After N iterations without seeing MAGIC, switches to "yield mode" (stop sending, wait). Alternates between yield and send.

**Assessment:**
- **Addresses Wormhole no-escape gap:** Even without `got_immediate_termination_signal()`, a timeout-based yield provides partial liveness — the sender stops consuming resources and can periodically retry.
- **Does NOT address deadlock per se:** With correct asymmetric roles, sender keeps sending until receiver replies. The only deadlock is when the peer never starts — and yield-then-retry doesn't fix that, it just makes the hang intermittent.
- **Risk of mutual yield deadlock:** If both sides yield simultaneously (possible with asymmetric roles if the receiver also has a yield mode), neither sends and both wait forever. Mitigation: only the sender yields, receiver always waits passively.
- **Better alternative exists:** Strategy G (enable termination signal on Wormhole) provides a clean host-controlled escape instead of firmware-level timeout heuristics.

**Verdict: Partial value — provides firmware-level resilience for Wormhole. But Strategy G is strictly superior if feasible. Consider as fallback if Strategy G is blocked.**

**Complexity:** Low — small firmware change in sender loop
**Wormhole concern:** Designed specifically for Wormhole's no-escape gap. Must carefully avoid mutual-yield.

### Strategy F: Canary Value + Compare (Read-Before-Write)

**Description:** Before entering handshake loop, read peer's `local_value` via ETH. If already non-zero (MAGIC from prior run or early peer), skip to ack.

**Assessment:**
- **No remote-read primitive:** `eth_send_packet` writes TO remote L1. There is no `eth_read_packet` that reads FROM remote L1. The ERISC can only read its own L1. To "read" the peer's state, the peer must send it — which is what the handshake already does.
- **Reading LOCAL `local_value` before loop:** This is already done — the while-loop condition checks `handshake_info->local_value != MAGIC_HANDSHAKE_VALUE`. If it's already MAGIC (from a prior session's stale write), the loop exits immediately. This is actually the stale-state bug, not a fix.
- **FIX HS1/HS2 already handles the case:** where one side finishes before the other starts — the final packet unblocks the late starter.

**Verdict: Rejected — not implementable (no remote read), and the local read variant is already part of the protocol (while-loop condition).**

**Complexity:** N/A — not feasible
**Wormhole concern:** N/A


### Strategy G: Remove ARCH_WORMHOLE Termination Signal Gap

**Description:** Enable `got_immediate_termination_signal()` check on Wormhole by removing the `#ifndef ARCH_WORMHOLE` guards.

**Assessment:**
- **Addresses the most critical gap:** Without termination signal support, Wormhole ERISCs in handshake have NO escape path. The host can only hard-reset them via `assert_risc_reset_at_core`, which doesn't reset ETH MAC state and leaves stale DMA operations.
- **Why was it disabled?** The `#ifdef` pattern appears throughout the fabric code. `got_immediate_termination_signal()` reads two L1 locations every iteration:
  1. `termination_signal_ptr` — a write from host
  2. `launch_msg->kernel_config.exit_erisc_kernel` — mailbox check

  On Wormhole, the mailbox check may have a hardware or performance concern — possibly related to the ERISC's L1 cache coherence model or the fact that Wormhole's ETH firmware uses `run_routing()` for context switching. The `run_routing()` call (present in both sender and receiver loops on Wormhole, lines 105–107 of `edm_handshake.hpp`) may conflict with the launch_msg mailbox read if the mailbox lives in a different L1 bank.

  Without seeing the original commit that added the `#ifdef`, the most likely reason is: **Wormhole ERISC base firmware does not set `exit_erisc_kernel` in the launch_msg in the same way Blackhole does**, making the check either incorrect or triggering false positives. The `run_routing()` mechanism is the Wormhole-specific context switch path.

- **Risk:** Enabling the check without understanding why it was disabled could cause:
  - False termination (ERISC exits handshake prematurely)
  - Hang (if the mailbox read blocks or returns stale data)
  - Performance regression (extra L1 reads per iteration)

- **Implementation path:** If `exit_erisc_kernel` is not set by Wormhole base firmware, only check `termination_signal_ptr` (host-written L1 word). This avoids the mailbox issue:

  ```cpp
  // Wormhole-safe termination check — only host L1 signal, no mailbox
  while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
         && *termination_signal_ptr != IMMEDIATELY_TERMINATE
  ) { ... }
  ```

**Verdict: Highest priority fix. The Wormhole termination gap is the single biggest reliability issue — it makes deadlocked handshakes unrecoverable without RISC hard reset. Should be investigated and fixed.**

**Complexity:** Low if only checking `termination_signal_ptr` (skip mailbox). Medium if full `got_immediate_termination_signal()` needs to be validated on Wormhole.
**Wormhole concern:** This IS the Wormhole fix. Must validate that `termination_signal_ptr` write from host is reliably visible to ERISC L1 read (cache coherence). The `invalidate_l1_cache()` call in the loop body should handle this.


## 4. Recommendation

### Tier 1: Fix Now (Highest Impact)

**Strategy G (partial): Enable termination signal check on Wormhole for `termination_signal_ptr` only.**

Rationale: The Wormhole no-escape gap is the most operationally impactful issue. Every deadlocked handshake on Wormhole requires a RISC hard reset, which corrupts ETH MAC state and cascades into the next session. A simple `termination_signal_ptr` check (without the full `got_immediate_termination_signal()` which includes the mailbox) provides a clean host-controlled escape path.

### Tier 2: Defense-in-Depth (Medium Priority)

**Strategy C (modified): Add a session nonce to the handshake.**

Instead of a monotonically incrementing sequence number (which requires persistent L1), use a **host-written session nonce**. Before launching ERISC firmware, the host writes a random 32-bit nonce to a reserved L1 word. The firmware includes this nonce in the handshake scratch buffer. The peer validates that the received nonce matches what the host told it to expect (also written to L1 before launch).

This eliminates the stale-RX-after-reset gap (Section 1.6) without requiring L1 persistence across resets. The host already writes compile-time args before launch — adding one more word is trivial.

### Tier 3: Keep As-Is (Already Correct)

- **Strategy A:** Already implemented (compile-time role assignment via chip_id tiebreak)
- **FIX HS1/HS2:** Correct defensive measure, keep in place
- **FIX AH:** Correct TX queue flush, keep in place

### Rejected

- **Strategy B (CSMA-CD):** Not implementable — no remote-read ETH primitive
- **Strategy D (host-mediated role):** Redundant with compile-time args
- **Strategy E (timeout-yield):** Inferior to Strategy G; consider only if G is blocked
- **Strategy F (canary read-before-write):** Not implementable — no remote-read ETH primitive


## 5. Implementation Plan: Strategy G (Wormhole Termination Signal)

### 5.1 Files to Modify

1. **`tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp`**
   - `fabric_sender_side_handshake()`: Replace `#ifndef ARCH_WORMHOLE` guard around termination check with a direct `*termination_signal_ptr` read (bypassing `got_immediate_termination_signal()` which includes the risky mailbox check)
   - `fabric_receiver_side_handshake()`: Same change
   - Post-loop final send: keep the `#ifndef ARCH_WORMHOLE` guard for the `got_immediate_termination_signal()` call in the post-loop send, OR use the simplified check

2. **`tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp`**
   - `sender_side_handshake()`: Consider adding a termination check here too, though the base handshake is used less frequently

3. **`tt_metal/fabric/hw/inc/tt_fabric_utils.h`**
   - `wait_for_notification()`: Same `#ifndef ARCH_WORMHOLE` pattern — apply same fix

### 5.2 Specific Changes

In `fabric_sender_side_handshake()`, replace:

```cpp
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
       && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#endif
) {
```

With:

```cpp
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
       && *termination_signal_ptr != tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE
) {
```

Or, more conservatively, add a Wormhole-specific simple check:

```cpp
while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
       && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#else
       && *termination_signal_ptr != tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE
#endif
) {
```

This preserves the existing Blackhole behavior while adding a minimal Wormhole escape path. The `invalidate_l1_cache()` call at the end of the loop body (already present) ensures the read of `termination_signal_ptr` sees host writes.

### 5.3 Post-Loop Final Send Guard

For the post-loop FIX HS2 final send, use the same simplified check on Wormhole:

```cpp
#ifndef ARCH_WORMHOLE
    if (!tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)) {
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
#else
    if (*termination_signal_ptr != tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) {
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
#endif
```

### 5.4 Host-Side Integration

The host already writes `IMMEDIATELY_TERMINATE` to `termination_signal_addr` during teardown (this is how Blackhole cooperatively terminates). On Wormhole, the same write path exists but the firmware ignores it. After this fix:

1. **Normal teardown:** Host writes `IMMEDIATELY_TERMINATE` → ERISC exits handshake loop → ERISC proceeds to termination → clean exit
2. **Deadlock recovery:** Host detects handshake stuck (status = `STARTED` for >5s) → writes `IMMEDIATELY_TERMINATE` → ERISC exits → host can safely proceed

### 5.5 Testing

- **Existing test coverage:** The racecondition-hunt branch has 76+ GAP tests that exercise fabric init/teardown. Run the full suite on Wormhole hardware after the change.
- **Specific regression test:** Force a handshake deadlock (e.g., delay one side's launch by 10s) and verify the host can terminate via `IMMEDIATELY_TERMINATE`. Verify the ERISC exits the handshake loop and reports `TERMINATED` status.
- **Performance test:** Measure AllGather bandwidth before/after — the extra `termination_signal_ptr` read per loop iteration should be negligible (one L1 read, ~1 cycle, already dwarfed by `eth_send_packet` which takes ~10s of cycles per TXQ interaction).

### 5.6 Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `termination_signal_ptr` contains stale IMMEDIATELY_TERMINATE from prior session | Low — host writes KEEP_RUNNING before launch | Verify host init writes `KEEP_RUNNING` to `termination_signal_addr` before ERISC deassert |
| L1 cache prevents ERISC from seeing host write | Very low — `invalidate_l1_cache()` called every iteration | Already mitigated by existing cache invalidation |
| Performance regression from extra read per iteration | Negligible — 1 L1 read vs ~20 register writes for eth_send_packet | Measure with bandwidth benchmark |


## 6. Implementation Plan: Strategy C (Session Nonce, Defense-in-Depth)

This is a follow-up optimization, lower priority than Strategy G.

### 6.1 Concept

Replace `MAGIC_HANDSHAKE_VALUE = 0xAA` with a per-session 32-bit nonce. The host generates a unique nonce for each firmware launch and writes it to both sides of the link before deassert.

### 6.2 Changes

1. **Host (`erisc_datamover_builder.cpp`):** Generate a 32-bit random nonce per link per session. Write it to `scratch[0]` location via compile-time arg or L1 write.
2. **Firmware (`edm_handshake.hpp`):** `init_handshake_info()` reads the host-provided nonce instead of using hardcoded `0xAA`. Loop condition checks for the expected nonce value instead of `MAGIC_HANDSHAKE_VALUE`.
3. **Firmware (`fabric_router_eth_handshake.hpp`):** Same changes.

### 6.3 Benefit

A stale in-flight ETH packet from a prior session would carry the OLD nonce. The new firmware instance expects the NEW nonce. The stale packet's value won't match → no false handshake completion.

### 6.4 Complexity

Low — one new compile-time arg, small firmware change. No host protocol changes needed.
