<!-- SUMMARY: Protocol design analysis comparing TCP handshake, RDMA QP init, and Two Generals problem against TT fabric ERISC ETH handshake deadlock
     KEYWORDS: tcp, handshake, deadlock, erisc, eth, fabric, init, symmetric, two-generals, rdma, protocol, FIX AE, FIX AF, FIX HS1, FIX HS2, simultaneous-open
     SOURCE: Code analysis of edm_handshake.hpp, fabric_router_eth_handshake.hpp, fabric_erisc_router.cpp, erisc_datamover_builder.cpp, device.cpp; AI-JOURNAL.md investigation history
     SCOPE: Diagnosis of the STARTED-STARTED symmetric deadlock in fabric init, comparison with TCP/RDMA/TLS protocols, ranked fix proposals
     USE WHEN: Debugging fabric init hangs where channels are stuck at STARTED (0xa0b0c0d0), designing new handshake protocols, evaluating fix proposals for #42429 -->

# TCP Handshake vs. TT Fabric ERISC ETH Handshake: Protocol Design Analysis

## Part 1: The TCP Three-Way Handshake as a Model

### 1.1 The SYN -> SYN-ACK -> ACK Sequence

TCP's connection establishment (RFC 793 Section 3.4) uses a three-message handshake:

```
  Client (Initiator)              Server (Listener)
  ──────────────────              ─────────────────
  CLOSED                          LISTEN
       ──── SYN (seq=x) ────►
  SYN-SENT                        SYN-RECEIVED
       ◄── SYN-ACK (seq=y, ack=x+1) ──
  ESTABLISHED                     SYN-RECEIVED
       ──── ACK (ack=y+1) ──►
  ESTABLISHED                     ESTABLISHED
```

**Key properties:**

1. **Asymmetric roles**: One side is a *client* (active open, sends SYN first) and one is a *server* (passive open, in LISTEN state). The roles are set before any message exchange.

2. **Unique state per direction**: Each side's state machine has different states at each point: `CLOSED->SYN-SENT->ESTABLISHED` vs `LISTEN->SYN-RECEIVED->ESTABLISHED`. They cannot get confused about who is in what role.

3. **Sequence numbers**: Each side's ISN (Initial Sequence Number) is independently chosen and acknowledged. This makes each message uniquely identifiable — retransmits are idempotent.

4. **The three messages form a logical chain**: SYN proves "I want to connect." SYN-ACK proves "I heard you AND I want to connect." ACK proves "I heard your acceptance." Each message references the previous one via ack numbers.

### 1.2 TCP Simultaneous Open (RFC 793 Section 3.4)

TCP *does* handle the case where both sides send SYN simultaneously (no server listening):

```
  Side A                          Side B
  ──────                          ──────
  CLOSED                          CLOSED
       ──── SYN (seq=x) ────►
  SYN-SENT                   ◄── SYN (seq=y) ────
                                  SYN-SENT
  SYN-RECEIVED (sees SYN from B while in SYN-SENT)
       ──── SYN-ACK (seq=x, ack=y+1) ────►
                              ◄── SYN-ACK (seq=y, ack=x+1) ────
  ESTABLISHED                     ESTABLISHED
```

**How it avoids deadlock:**
- When a SYN arrives while in SYN-SENT (not LISTEN), TCP transitions to SYN-RECEIVED and sends a SYN-ACK. This is a *different* transition than the normal case.
- The state machine explicitly has a `SYN-SENT + receive SYN -> SYN-RECEIVED` transition.
- Both sides converge to ESTABLISHED after exchanging SYN-ACKs.

**Critical insight**: TCP handles simultaneous open because the state machine has *explicit transitions for every possible message arrival in every possible state*. There is no "undefined" state combination.

### 1.3 Reliable Delivery Mechanisms

TCP achieves reliable handshake over an unreliable channel via:

1. **Retransmission with exponential backoff**: If SYN gets no SYN-ACK within a timeout, retransmit.
2. **Idempotent messages**: Receiving a duplicate SYN in SYN-RECEIVED is harmless — just re-send SYN-ACK.
3. **Sequence numbers**: Distinguish stale/retransmitted messages from new ones.
4. **Timeout + abort**: After N retries, give up and reset to CLOSED. Deadlock is impossible because both sides will eventually timeout independently.

### 1.4 Parallel to Fabric Init

| TCP Concept | Fabric Init Analogue |
|---|---|
| SYN | `eth_send_packet(scratch -> remote.local_value)` with MAGIC_HANDSHAKE_VALUE |
| SYN-ACK | Receiver sees MAGIC in local_value, sends MAGIC back to sender's local_value |
| Client/Server roles | `is_handshake_sender` (compile-time, based on chip_id tiebreak) |
| Unreliable channel | ETH link DMA — fire-and-forget, no ACK at the packet level |
| Connection state | `edm_status` register: STARTED -> REMOTE_HANDSHAKE_COMPLETE -> LOCAL_HANDSHAKE_COMPLETE -> READY_FOR_TRAFFIC |
| ISN / Sequence number | **None** — only MAGIC_HANDSHAKE_VALUE (0xAA), no unique per-session identifier |
| Retransmission | Sender loops sending MAGIC until it sees MAGIC in its own local_value |
| Timeout / abort | Only at the *host* level (kStartedTimeoutMs), not in firmware |


## Part 2: Other Relevant Protocol Designs

### 2.1 RDMA Queue Pair Initialization (IB/RoCE MODIFY_QP)

InfiniBand QP state transitions: `RESET -> INIT -> RTR -> RTS`.

```
  Side A (Client)               Side B (Server)
  ──────────────                ──────────────
  QP in RESET                   QP in RESET
  MODIFY_QP -> INIT             MODIFY_QP -> INIT
  Exchange QPN, GID, LID via out-of-band channel (e.g., TCP socket)
  MODIFY_QP -> RTR              MODIFY_QP -> RTR
  (A can now RECEIVE)           (B can now RECEIVE)
  MODIFY_QP -> RTS              MODIFY_QP -> RTS
  (A can now SEND)              (B can now SEND)
```

**Why no deadlock:**
- The QP state transitions are *local* operations (MODIFY_QP is a verb call to the local HCA). They do not depend on the remote side being in any particular state.
- The QPN/address exchange happens via a *separate, reliable channel* (typically a TCP socket). The RDMA channel is not used to bootstrap itself.
- RTR means "ready to receive" — a side can receive before the peer can send. This temporal asymmetry is designed in.

**Lesson for fabric init**: Using the ETH link to bootstrap the ETH link's own initialization creates a circular dependency. RDMA avoids this by using an out-of-band channel for the bootstrap.

### 2.2 TLS 1.3 Handshake

```
  Client                        Server
  ──────                        ──────
  ClientHello (+ key share) ──►
                            ◄── ServerHello (+ key share)
                            ◄── {EncryptedExtensions}
                            ◄── {Certificate}
                            ◄── {CertificateVerify}
                            ◄── {Finished}
  {Finished} ──►
  Application Data ◄──►         Application Data
```

**How asymmetric roles help:**
- Client always sends first. Server always responds.
- No simultaneous-open scenario exists — the protocol literally cannot start from both sides.
- The asymmetry is established by convention/configuration (who connects to whose port).

**Lesson**: When you can *guarantee* one side starts first, you eliminate an entire class of protocol bugs. The fabric init tries to do this via host-side launch ordering (FIX AE), but the ordering is not ironclad.

### 2.3 LLDP / 802.3 Link Detection

LLDP (Link Layer Discovery Protocol, IEEE 802.1AB) is purely passive and symmetric:

- Both sides periodically transmit LLDP frames (every 30s by default).
- Both sides independently listen for LLDP frames from the peer.
- There is no handshake, no state machine dependency, no ACK.
- A side considers the link "up" when it receives an LLDP frame from the peer.

**Why it doesn't deadlock:**
- There is *no dependency between the two directions*. A's decision to send does not depend on B's state, and vice versa.
- The protocol is convergent: given enough time, both sides will have received a frame from the other.
- Idempotent: receiving a duplicate frame is a no-op.

**Lesson**: The fabric handshake deadlocks because each side's ability to *progress* depends on the other side's state. LLDP avoids this by making each side's actions independent. A fabric protocol where both sides send unconditionally (like LLDP) and poll for the peer's arrival independently would be deadlock-free.

### 2.4 The Two Generals' Problem

The Two Generals' Problem (Akkoyunlu, Ekanadham, Huber, 1975) proves that **no protocol can guarantee coordinated action between two parties over an unreliable channel using a finite number of messages**.

However, our problem is subtler:
- We don't need *guaranteed* coordination in finite messages.
- We need *eventual* coordination given that the channel is actually reliable (ETH DMA succeeds within bounded time when both endpoints are alive).
- The Two Generals' impossibility applies to *unreliable* channels. The ETH link is reliable when both ERISCs are running — packets are not lost in transit.

**What the Two Generals' Problem DOES say about our situation:**
- If one side's `init_handshake_info()` can erase the other side's already-delivered message (by zeroing `local_value`), then the channel is effectively *unreliable at the application layer* even though it is reliable at the link layer. This is the root of the FIX HS1/HS2 race.
- The `init_handshake_info()` reset creates an artificial message-loss window. Eliminating that window (or making messages idempotent across it) is sufficient to solve the problem.


## Part 3: Diagnosis of Current Protocol

### 3.1 Current Handshake Protocol

**Role assignment (compile-time):**
- `is_handshake_master = (local_chip_id < peer_chip_id)` within same mesh, or `(local_mesh_id < peer_mesh_id)` across meshes.
- Assigned in `erisc_datamover_builder.cpp:890`.
- Master = sender role. Subordinate = receiver role.

**Sender-side protocol (`sender_side_handshake`):**
```
1. init_handshake_info():
   a. If ETH TXQ busy, flush it (FIX AH)
   b. Set local_value = 0
   c. Set scratch[0] = MAGIC (0xAA)
   d. Set scratch[1] = (my_mesh_id | my_device_id << 16)
2. Loop:
   a. If local_value == MAGIC → exit loop
   b. eth_send_packet(scratch -> remote.local_value)  // fire-and-forget
   c. invalidate_l1_cache()
3. Post-loop: eth_send_packet(scratch -> remote.local_value) once more (FIX HS1/HS2)
```

**Receiver-side protocol (`receiver_side_handshake`):**
```
1. init_handshake_info():  (same as sender)
2. Loop:
   a. If local_value == MAGIC → exit loop
   b. (NO send — just poll)
   c. invalidate_l1_cache()
3. Post-loop: eth_send_packet(scratch -> remote.local_value) once (the ACK)
```

**State progression after handshake:**
```
edm_status = STARTED                        (before handshake)
edm_status = REMOTE_HANDSHAKE_COMPLETE      (after ETH handshake)
[local sync between ERISCs on same chip]
edm_status = LOCAL_HANDSHAKE_COMPLETE       (all local ERISCs synced)
[wait for host to write READY_FOR_TRAFFIC]
edm_status = READY_FOR_TRAFFIC              (host confirms)
```

### 3.2 Where the Symmetric Deadlock Occurs

The deadlock occurs when **both sides execute `sender_side_handshake` simultaneously** (or both execute `fabric_sender_side_handshake`). This happens despite the compile-time role assignment because:

1. **Host-side launch timing is the real role enforcer, not the compile-time flag.** The compile-time `is_handshake_sender` flag determines which *function* each side calls. But the protocol only works correctly when the receiver is already in its poll loop *before* the sender starts sending.

2. **The actual deadlock scenario (from AI-JOURNAL, FIX AE comment):**
   - Device 4 (non-MMIO, slow relay ~200ms/channel) gets its ETH cores launched.
   - Device 5 (non-MMIO, fast relay <1ms via idle MMIO partner) gets its ETH cores launched ~6ms later.
   - Both devices' ERISCs enter their handshake functions within ~6ms of each other.
   - The "sender" side sends MAGIC to the peer's `local_value`.
   - The "receiver" side's `init_handshake_info()` **zeros its own `local_value`**, erasing the sender's already-delivered MAGIC.
   - Now the sender has exited (it saw MAGIC in its own local_value from the other side's sends during init), but the receiver's local_value is 0 and nobody is sending to it anymore.
   - **Deadlock**: receiver polls forever, sender has exited.

3. **FIX HS1/HS2 partially addressed this** by having the sender send one final packet after exiting the loop. But this only works if the receiver's `init_handshake_info()` has *already completed* before the sender exits. If the receiver is still in `init_handshake_info()` when the final packet arrives, that packet will also be erased.

### 3.3 What Property the Current Protocol Violates

**The current protocol violates the property of *atomic role transition*.**

In TCP:
- A server in LISTEN state does not modify any shared state until it receives a SYN. The SYN reception is what triggers the transition to SYN-RECEIVED. The server's role as "responder" is enforced by the fact that it *waits* before doing anything.
- A client in SYN-SENT does not zero out its receive buffer — it just watches for incoming packets.

In the fabric protocol:
- Both sides call `init_handshake_info()` which zeros `local_value`. This is a *destructive write to shared state* that can erase an already-delivered message from the peer.
- The receiver doesn't just passively wait — it actively destroys state as part of its initialization.
- There is no equivalent of TCP's "don't touch the receive buffer" discipline.

**This is a "confused init" scenario** — neither pure "simultaneous open" nor "lost ACK." The closest TCP analogue is if both sides sent a RST to each other during simultaneous open, zeroing out the received SYN. TCP explicitly does NOT do this.

### 3.4 Classification

| Property | TCP | Fabric Init |
|---|---|---|
| Roles assigned before first message? | Yes (client/server) | Yes (sender/receiver via chip_id) |
| Role determines behavior? | Yes (different state machines) | Yes (sender sends+polls, receiver polls+acks) |
| Init can destroy peer's delivered message? | No | **YES** (`init_handshake_info` zeros `local_value`) |
| Explicit simultaneous-open handling? | Yes (SYN-SENT + recv SYN -> SYN-RECEIVED) | **No** — sender assumes receiver is already polling |
| Timeout at protocol level? | Yes (retransmit timer) | **No** — firmware loops forever; only host has timeout |
| Idempotent message delivery? | Yes (duplicate SYN in SYN-RECEIVED is harmless) | **No** — MAGIC write can be erased by concurrent init |


## Part 4: Concrete Proposed Fixes (Ranked)

### Evaluation Criteria (from TCP design principles):

1. **Asymmetric roles?** — Does one side start first by construction?
2. **Handles simultaneous entry?** — What happens if both sides start within the init window?
3. **Handles loss/timeout?** — Does the protocol converge without host intervention?
4. **Minimum state?** — What does each side need to track?
5. **Implementation complexity** — Lines of code, risk of regression.

---

### Fix A: Eliminate the Destructive Init (Recommended)
**Concept**: Separate `init_handshake_info` from the handshake loop. Zero `local_value` during TXQ/stream-register init (which happens long before the handshake), NOT inside the handshake function. By the time the handshake loop starts, `local_value` is already zeroed and has been for hundreds of microseconds.

```
CURRENT:
  init_handshake_info():
    flush TXQ       ← safe, do early
    local_value = 0  ← DANGEROUS: destroys peer's delivered MAGIC
    scratch = MAGIC  ← safe
  sender_loop: send + poll
  receiver_loop: poll + ack

PROPOSED:
  [during Object Setup, before STARTED]
    local_value = 0
    scratch = MAGIC
    flush TXQ
  [edm_status = STARTED]
  [... Object Setup continues ...]
  sender_loop: send + poll  (no init inside loop)
  receiver_loop: poll + ack (no init inside loop)
```

| Criterion | Rating | Notes |
|---|---|---|
| Asymmetric roles | N/A | Not needed — eliminates the root cause directly |
| Simultaneous entry | **Handled** | Both sides can send/poll simultaneously without erasing each other's messages |
| Loss/timeout | Sender loops until peer responds; if peer never starts, host timeout fires | Same as current |
| Minimum state | `local_value` + `scratch` (unchanged) | No new state |
| Complexity | **Low** | Move 3 lines of init earlier in the init sequence |

**TCP parallel**: This is equivalent to TCP's rule that SYN-RECEIVED does not zero the receive buffer. The receive side's initialization is done *before* it enters the LISTEN state, not concurrently with message arrival.

**Risk**: Must ensure `local_value` stays zero between the early init and the handshake loop. No other firmware code should write to it in between. Review the Object Setup phase for accidental overwrites.

---

### Fix B: Host-Mediated Launch Ordering (Current FIX AE + FIX AF approach)
**Concept**: The host ensures MMIO devices launch their ETH ERISCs before non-MMIO devices, and non-MMIO devices are launched sequentially with a barrier (`wait_for_eth_cores_launched`) between each.

```
  Host: launch MMIO ETH (fast, direct PCIe)
  Host: launch non-MMIO device 4 (slow, relay)
  Host: wait for device 4 ERISCs to reach STARTED
  Host: launch non-MMIO device 5 (slow, relay)
  Host: wait for device 5 ERISCs to reach STARTED
  ...
```

| Criterion | Rating | Notes |
|---|---|---|
| Asymmetric roles | **Yes** — MMIO always first, non-MMIO sequential | Good |
| Simultaneous entry | **Partially handled** — depends on timing margins | FIX AE's 200ms relay gap helps but is not guaranteed |
| Loss/timeout | Host timeout at kStartedTimeoutMs | Same as current |
| Minimum state | None new on firmware side | Complexity is on host side |
| Complexity | **Medium-High** | Already implemented across device.cpp + mesh_device.cpp; complex ordering logic |

**TCP parallel**: This is like guaranteeing the server is in LISTEN before the client sends SYN. It works, but depends on timing guarantees that may not hold under load or during quiesce (where the relay path introduces variable latency).

**Weakness**: Does not protect against the `init_handshake_info` erase race when both sides start within the ~6ms window. The root cause (destructive init) remains.

---

### Fix C: Make Sender-Side Handshake Retry-Safe (Post-Exit Burst)
**Concept**: After the sender exits its loop, send N packets (not just 1) over a time window long enough to guarantee the receiver's `init_handshake_info` has completed.

```
  sender exits loop:
    for (i = 0; i < 100; i++) {
      eth_send_packet(scratch -> remote.local_value);
      // ~10ns per send, 100 sends = ~1us burst
    }
```

| Criterion | Rating | Notes |
|---|---|---|
| Asymmetric roles | Yes (existing chip_id tiebreak) | Unchanged |
| Simultaneous entry | **Better than FIX HS1** — burst covers a wider timing window | But not guaranteed if receiver's init takes >1us |
| Loss/timeout | Same as current | |
| Minimum state | None new | |
| Complexity | **Low** | Change 1 line to a loop |

**Weakness**: Still a timing band-aid. If the receiver's `init_handshake_info` takes longer than the burst duration (e.g., TXQ flush takes variable time), the race remains.

---

### Fix D: LLDP-Style Bidirectional Unconditional Send
**Concept**: Abolish the sender/receiver distinction entirely. Both sides enter the same function: send MAGIC and poll for MAGIC, simultaneously. Neither side needs to wait for the other.

```
  both_sides_handshake():
    init_handshake_info()
    loop:
      eth_send_packet(scratch -> remote.local_value)
      if local_value == MAGIC: exit
      invalidate_l1_cache()
    // post-exit: one final send (FIX HS pattern)
    eth_send_packet(scratch -> remote.local_value)
```

| Criterion | Rating | Notes |
|---|---|---|
| Asymmetric roles | **No** — both sides are symmetric | Intentionally symmetric (like LLDP) |
| Simultaneous entry | **Fully handled** — both sides send unconditionally | |
| Loss/timeout | **Handled** — both sides keep sending until success | |
| Minimum state | Same as current | |
| Complexity | **Very Low** — replace sender/receiver functions with one | Actually simplifies the code |

**TCP parallel**: This is TCP's simultaneous-open case, but done *intentionally* rather than as a corner case. Both sides are in "SYN-SENT" and both send SYN-ACKs when they see the peer's SYN.

**Critical issue**: The `init_handshake_info` erase race STILL exists. If side A sends MAGIC to B.local_value, and then B's `init_handshake_info` zeros B.local_value, A's message is lost. B will eventually re-send MAGIC to A, A sees it and exits, but B is stuck because A stopped sending.

**Mitigation**: Combine with Fix A (early init). If `local_value` is zeroed during Object Setup (before STARTED), not inside the handshake function, then both sides enter the send-and-poll loop with `local_value = 0` and no destructive init can erase a peer's message.

---

### Fix E: Replace ETH DMA Pre-Ping with Host-Mediated Signal
**Concept**: Don't use the ETH link at all for the handshake coordination. Instead, the host writes a "peer is ready" flag to each ERISC's L1 via PCIe (for MMIO devices) or relay (for non-MMIO devices). The ERISC firmware polls this host-written flag instead of waiting for an ETH DMA packet from the peer.

| Criterion | Rating | Notes |
|---|---|---|
| Asymmetric roles | Yes — host is the trusted coordinator | Like RDMA's out-of-band QPN exchange |
| Simultaneous entry | **Cannot occur** — host serializes the signaling | |
| Loss/timeout | PCIe/relay writes are reliable from host perspective | |
| Minimum state | One flag per ERISC | |
| Complexity | **High** — requires host-side coordination logic, relay writes to non-MMIO | |

**RDMA parallel**: This is exactly how RDMA QP initialization works — the QPN exchange happens over a reliable out-of-band channel (TCP socket), not over the RDMA link itself. The RDMA link is not used to bootstrap itself.

**Weakness**: Adding host-mediated relay writes for every non-MMIO ERISC handshake adds latency. Relay writes are slow (~200ms per channel per non-MMIO device). For a T3K with 4 non-MMIO devices x 4 channels = 16 relay writes, this adds seconds to fabric init.

---

### Ranking (Most to Least TCP-Sound)

```
Rank  Fix   TCP Soundness  Implementation Risk  Notes
────  ────  ─────────────  ──────────────────  ─────
 1    A+D   ★★★★★          Low                 Eliminate destructive init + LLDP-style symmetric send.
                                                Solves the root cause (init erase) AND handles
                                                simultaneous entry. Simplifies firmware.
 2    A     ★★★★☆          Low                 Eliminate destructive init alone. Requires correct
                                                sender/receiver timing (existing FIX AE ordering),
                                                but eliminates the erase window.
 3    E     ★★★★☆          High                Host-mediated signal (RDMA-style). Correct by
                                                construction but adds latency and complexity.
 4    B     ★★★☆☆          Medium              Current FIX AE/AF approach. Timing-dependent.
                                                Works in practice but not provably correct.
 5    C     ★★☆☆☆          Low                 Burst sends. Band-aid over the timing race.
                                                Not fundamentally sound.
```


## Part 5: Recommended Implementation

### Recommended Fix: A+D (Eliminate Destructive Init + Symmetric Send)

This combines the two best properties:
1. **Fix A**: Move the destructive `local_value = 0` and `scratch = MAGIC` init out of the handshake function and into the Object Setup phase (before `edm_status = STARTED`). This eliminates the message-erase window.
2. **Fix D**: Replace the sender/receiver handshake functions with a single symmetric function where both sides send and poll. This eliminates the need for timing-based launch ordering.

### Protocol Flow (RFC-style Sequence Diagram)

```
  ERISC A (chip 0)                        ERISC B (chip 4)
  ────────────────                        ────────────────

  [Object Setup]                          [Object Setup]
  local_value = 0                         local_value = 0
  scratch[0] = MAGIC                      scratch[0] = MAGIC
  scratch[1] = my_identity                scratch[1] = my_identity
  flush TXQ if busy                       flush TXQ if busy
  edm_status = STARTED                    edm_status = STARTED

  [... remaining Object Setup ...]        [... remaining Object Setup ...]

  [Enter symmetric handshake]             [Enter symmetric handshake]
  ┌─────────────────────────────┐         ┌─────────────────────────────┐
  │ loop:                       │         │ loop:                       │
  │   send scratch->B.local_val │────────►│   (B.local_value = MAGIC)   │
  │   if A.local_val == MAGIC:  │         │   send scratch->A.local_val │
  │     break                   │◄────────│   if B.local_val == MAGIC:  │
  │   invalidate_l1_cache()     │         │     break                   │
  │   (context switch if due)   │         │   invalidate_l1_cache()     │
  └─────────────────────────────┘         └─────────────────────────────┘

  // Post-loop: one final send
  send scratch->B.local_val ────────────► (idempotent, B already exited or
                                           still in loop — either way safe)
                              ◄────────── send scratch->A.local_val

  edm_status = REMOTE_HANDSHAKE_COMPLETE  edm_status = REMOTE_HANDSHAKE_COMPLETE

  [local ERISC sync]                      [local ERISC sync]
  edm_status = LOCAL_HANDSHAKE_COMPLETE   edm_status = LOCAL_HANDSHAKE_COMPLETE

  [host writes READY_FOR_TRAFFIC]         [host writes READY_FOR_TRAFFIC]
```

**Why this is deadlock-free:**
- Both sides send unconditionally. If A starts before B, A's sends arrive at B.local_value. When B starts, B.local_value is NOT zeroed (init happened earlier), so B sees MAGIC immediately and exits.
- If both sides start simultaneously, they send to each other in parallel. Both see MAGIC and exit.
- The FIX HS1/HS2 post-loop send is retained as defense-in-depth, but is no longer required for correctness.
- No message can be erased because `init_handshake_info` no longer exists as a separate step during the handshake.

### Code Locations to Change

**1. `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp`**
- Refactor `init_handshake_info()` to NOT be called inside `sender_side_handshake` / `receiver_side_handshake`.
- Create a new `prepare_handshake_state()` function that only does the TXQ flush + local_value zero + scratch init.
- Replace `sender_side_handshake` and `receiver_side_handshake` with a single `symmetric_handshake()` that only does the send+poll loop.

**2. `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp`**
- Same refactor for `fabric_sender_side_handshake` and `fabric_receiver_side_handshake` — replace with `fabric_symmetric_handshake()`.

**3. `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`**
- Around line 3249 (after `edm_status = STARTED`, during Object Setup): call `prepare_handshake_state(handshake_addr, mesh_id, device_id)`.
- Around line 3572-3587: replace the `if constexpr (is_handshake_sender)` branch with a single call to `fabric_symmetric_handshake()`.

**4. `tt_metal/fabric/erisc_datamover_builder.cpp`**
- Line 890: `is_handshake_master` computation can be removed (or retained for `IS_LOCAL_HANDSHAKE_MASTER` which is a separate concept for the local-ERISC sync).
- Line 1134: `IS_HANDSHAKE_SENDER` compile-time arg can be removed or repurposed.

**5. `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp`**
- Line 128: `is_handshake_sender` can be removed if the symmetric handshake is used unconditionally.

**6. Host-side simplification (optional, but recommended):**
- `tt_metal/impl/device/device.cpp` lines 1474-1506 (FIX AE defer logic): Can be simplified — the launch ordering constraint is no longer required for correctness. The deferred-launch mechanism can remain for performance (avoid relay congestion) but is no longer a correctness requirement.
- `tt_metal/impl/device/device.cpp` lines 2320-2345 (FIX AF `wait_for_eth_cores_launched`): No longer required for correctness. Can be retained as optional barrier for debug visibility.


## Appendix: Why FIX HS1/HS2 Is Necessary But Insufficient

FIX HS1 and FIX HS2 (the post-loop final send) address a specific sub-case of the race:

```
Timeline:
T=0:     A starts sender_side_handshake, calls init_handshake_info (A.local_value=0)
T=0.1ms: A enters send loop, sends MAGIC to B.local_value
T=6ms:   B starts sender_side_handshake (both in sender mode due to compile-time assignment)
T=6ms:   B calls init_handshake_info → B.local_value = 0 (erases A's MAGIC!)
T=6.1ms: B enters send loop, sends MAGIC to A.local_value
T=6.2ms: A sees MAGIC in A.local_value, exits loop
T=6.2ms: A sends one final packet (FIX HS1) → B.local_value = MAGIC
T=6.3ms: B sees MAGIC, exits
```

FIX HS1 works HERE because B's `init_handshake_info` has already completed by T=6.2ms. But consider:

```
Timeline (tighter race):
T=0:     A starts, sends MAGIC to B.local_value
T=5.9ms: B starts, calls init_handshake_info:
         - TXQ flush starts (variable time)
T=6.0ms: A sees MAGIC from B's sends, exits loop
T=6.0ms: A sends final packet (FIX HS1) → B.local_value = MAGIC
T=6.1ms: B's init_handshake_info finishes → B.local_value = 0 (erases FIX HS1 packet!)
T=6.1ms: B enters poll loop, local_value = 0, nobody sending → DEADLOCK
```

The FIX HS1 packet was erased because B's `init_handshake_info` hadn't finished yet. This window is small (~100ns for the zeroing), but on a 1GHz RISC-V core running tight loops with ETH DMA, it is plausible.

**Fix A eliminates this window entirely** by moving the zeroing to Object Setup, hundreds of microseconds before the handshake loop.
