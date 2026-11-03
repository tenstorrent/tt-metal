# Route-word preload (N/S fix 2) and the strict flush-defer: design note

Status: design for the user's review (2026-11-02). **Study A/B on the drafts: §8. The preload, as drafted, is flat or
costs on every measured throughput row.** Nothing here is built for adoption and nothing has run on a device.
The static numbers come from offline draft builds on v4rxg3 (§3), made only to count instructions and registers. Labels:
code = read from code; static = from the draft builds; meas = measured on a device; hyp = hypothesis.

User decisions this note follows (2026-11-02):
- Stage 1 preloads the route word only. The header word (payload size, send type, source channel) is a later,
  separate arm, built only if stage 1's profile shows its load is next on the path.
- An additive codec entry in `hostdevcommon/fabric_common.h` decodes from route bytes already loaded. The pointer form
  becomes load-then-decode, so the codec still owns the rule and its volatile loads. It changes only in the variant's
  copy of the file, never in the live tree.
- The strict flush-defer is designed here too and built after the preload lands.
- The note comes first; nothing is built for adoption before the user's review.

Background: V4_NS_FLOOR.md §2-§3 (why N/S routers sit at ~172 on the Galaxy), FLUSH_DEFER_DESIGN.md (the flush-defer
audit and the user's clarification), V4_RECEIVER_OWNERSHIP.md §18 (the placement rule this follows).

## 0. Summary

- **What S1 does.** At the end of a channel's step, if the next slot holds a packet already counted, its route is
  loaded. The raw value is kept in the channel's progress and decoded at the next step's admission. Nothing is
  removed; the loads move.
- **Why it can help, and only how (hyp).** The ERISC is in-order: a stalled load shows at its first use. The preload
  helps only if independent work runs between the load and that first use. Here that work is the rest of the step
  (completions), the other channels' steps, the frame and the next pass's reads.
- **First static finding: byte loads don't work in 2D.** GCC follows each volatile byte load (`lbu`) of a route byte
  with a redundant zero-extension (`zext.b` / `andi 0xff`), one or two instructions later. That extension is the
  load's first use, so the stall stays where the load is issued.
  - Seen in the drafts and in a minimal test compile (§3.2).
  - Today's code has the same pair; today it is harmless because the decode follows at once anyway.
  - A volatile word load (`lw`) has no extension.
  - With word loads the first use moves to the decode, 9 or more instructions later and usually in the next pass
    (static).
  - 1D is already a word (`routing_fields.value`) and works either way.
- **Second static finding: the extra carried values don't fit everywhere.**
  - Meshes have room: bh_2d, glx_2d and one ERISC pass the headroom check with the byte form, and nearly all pass with
    the word form (24/28, 44/48).
  - Rings, tori, VC1 split routers and the Galaxy express configs don't: hot values move to the stack (`txq_status`,
    `uncredited`, deliver's values). This matches headroom_f4.txt, where one synthetic value already fails on rings and
    tori.
- **Counted work.** S1 removes no work and adds about +4 to +8 instructions per forwarded packet (static, §3.5). Any
  gain must come from the stall it overlaps, which this note does not estimate (CLAUDE.md). The predictions in §4
  are stated that way.
- **Proposal.** The word form, for channels whose router passes the headroom check (meshes, the motivating N/S case).
  The scope rule is the user's choice (§7, Q1). Then the strict flush-defer (S3) on the same mechanism.

## 1. Code shape

### 1.1 The codec entry (hostdevcommon/fabric_common.h, the variant's copy only)

Additive. The pointer forms become load-then-decode, so their existing users keep the same code. Checked statically:
v4rxrp0 (v4rxg3 + this file) is byte-identical to v4rxg3 on all 2,863 configs at -O3. At -Os, 32 images differ,
none by an instruction: 12 only move, and 20 exchange two callee-saved registers (§3.1).

```cpp
// Two steps, which a router may separate: load_route_bytes reads the bytes it decodes from (volatile: the link's DMA
// writes them; the loads stay where the caller issues them), decode_action selects the action from bytes already
// loaded. The pointer forms do both at once.
struct RouteBytes { std::uint8_t y; std::uint8_t x; };            // the action bytes a router decodes from
template <eth_chan_directions MY_DIR>
static constexpr bool decodes_x_only = MY_DIR == EAST || MY_DIR == WEST;
template <bool X_ONLY>
static RouteBytes load_route_bytes(const volatile std::uint8_t* route_buffer, std::uint32_t local_y,
                                   std::uint32_t local_x, std::uint32_t y_size);
static constexpr std::uint8_t decode_action_y_first(RouteBytes bytes);   // y != 0 ? y : x
template <eth_chan_directions MY_DIR>
static constexpr std::uint8_t decode_action(RouteBytes bytes);           // x alone for E/W, else Y first
// pointer forms (unchanged signatures): load_route_bytes, then decode_action
```

**The word form (proposed for adoption; see §0 and §3.2).** The load step reads the 32-bit route-buffer words that
hold the router's Y and X bytes. The decode extracts the bytes:

```cpp
struct RouteWords { std::uint32_t y; std::uint32_t x; };   // raw: a volatile word load needs no extension
template <bool X_ONLY>
static RouteWords load_route_words(const volatile RouteBufferWord* route_words, std::uint32_t local_y,
                                   std::uint32_t local_x, std::uint32_t y_size);   // X_ONLY: x's word only
template <eth_chan_directions MY_DIR>
static constexpr std::uint8_t decode_action(RouteWords words, std::uint32_t local_y, std::uint32_t local_x,
                                            std::uint32_t y_size);   // (word >> 8 * (index & 3)) & 0xFF, then as above
static constexpr std::uint8_t decode_action_y_first(RouteWords words, std::uint32_t local_y, std::uint32_t local_x,
                                                    std::uint32_t y_size);
```

- The draft reads the words through a `reinterpret_cast` of the byte array, for static counts only.
- The build has no `-fno-strict-aliasing`, so the adoption form needs a word view of the route buffer from the header
  type (`HybridMeshPacketHeaderT`, a union member). That is a non-fabric file, pinned in the variant. Open question Q2.
- The route buffer starts 48 bytes into the 16-byte-aligned header (offset 48 in the code), so its words are aligned.

### 1.2 The router side (router_routing.hpp)

```cpp
// fabric_router: a packet's route as loaded, before the decode
#if defined(FABRIC_2D)
using RouteBytes = Routing2DCodec::RouteWords;   // (or ::RouteBytes, the byte form)
#else
using RouteBytes = uint32_t;                     // the 1D routing word
#endif

struct Routing2D<VC, INTERMESH_INGRESS> {
    static constexpr bool x_only = !INTERMESH_INGRESS && Routing2DCodec::decodes_x_only<MY_DIR>;
    // Landed first on an ingress channel (out of line), then the codec's volatile loads.
    static RouteBytes load(tt_l1_ptr PACKET_HEADER_TYPE* header, const ChipCoordinates& my_chip);
    // The route word from loaded bytes: the codec's decode (Y first after a landing); on an edge router the exit check
    // with the destination mesh (read here, at use; §1.6).
    static uint32_t route_word(RouteBytes bytes, tt_l1_ptr PACKET_HEADER_TYPE* header, const ChipCoordinates& my_chip);
};
struct Routing1D {
    static RouteBytes load(header, my_chip);              // a volatile load of routing_fields.value
    static uint32_t route_word(RouteBytes word, ...);     // the word as it is
};
```

`Routing::read` goes away; receive calls `load` (when nothing is preloaded) and `route_word`.

### 1.3 The progress field and the step (router_receiver.hpp)

```cpp
struct ReceiverChannelProgress {
    uint32_t packets_ready, next_slot_addr, next_index, uncredited, idle_passes_with_partial_group;
    // The next slot's route, loaded ahead: valid whenever packets_ready was nonzero at the step's start.
    RouteBytes next_route{};
};

template <size_t VC, typename Connections>
FORCE_INLINE void step(ReceiverChannelProgress& progress, ..., uint32_t arrived_raw, const ChipCoordinates& my_chip) {
    const uint32_t arrived = arrived_raw & stream_count_mask;
    const bool route_loaded = progress.packets_ready != 0;   // a packet left from an earlier pass: its route preloaded
    if constexpr (ReceiverChannel<VC>::first_level_ack) {
        credit_frame_due |= accumulate_first_level_acks<VC>(progress, credits, arrived);
    }
    const bool handled_packet = receive<VC>(progress, sources, downstreams, arrived, route_loaded, my_chip);
    // The next packet's route, loaded now when it is known present, kept after the completions, which run while the
    // loads complete.
    RouteBytes next_route = progress.next_route;               // kept unless loaded now
    if constexpr (ReceiverChannel<VC>::forwards) {
        if (handled_packet && progress.packets_ready != 0) {
            next_route = Routing::load(slot_header(progress.next_slot_addr), my_chip);
        }
    }
    credit_frame_due |= credit_completions<VC>(progress, sources, handled_packet, credits);
    progress.next_route = next_route;
}

// receive, on a forwarding channel:
//   if (!route_loaded) progress.next_route = Routing::load(header, my_chip);   // first packet after an idle spell
//   route_word = Routing::route_word(progress.next_route, header, my_chip);
```

- **Placement follows §18.** For `loop_local_vc` the field is part of the loop copy, so it is carried (in registers if
  the compiler can, which the headroom check confirms). For later VCs and on one ERISC it is in memory, and its store
  comes after the completions.
- **Join, not a second branch.** The unconditional store after a join avoids a second `if (preload)` around the
  completions. The first draft (v4rxrp1) had one, and GCC duplicated the completion and atomic-delivery code into both
  arms: loop +57 on 1D against +27 for the join form (static).
- **No valid flag; one invariant:** if `packets_ready` > 0 when a step starts, `next_route` holds the next slot's route.
  - Kept by: the preload when a step retires a packet and more are ready; receive storing what it loaded now when the
    packet then waits (admission failed); no change while a packet waits.
  - The first packet after idle (packets_ready was 0, refilled from this pass's count or this step's acks) takes
    today's path: load now.
- **Names (proposed, channel-kind rule).**
  - `ReceiverChannelProgress::next_route`.
  - Codec: `Routing2DCodec::RouteWords` (or `RouteBytes`), `load_route_words` (`load_route_bytes`),
    `decode_action(RouteWords, ...)`, `decodes_x_only<MY_DIR>`.
  - `fabric_router::RouteBytes`; `Routing2D::load` / `route_word`, `Routing1D::load` / `route_word`.

### 1.4 The strict flush-defer record and gate (S3; designed here, built after S1)

The user's clarification (FLUSH_DEFER_DESIGN.md, 2026-11-02) is the rule:
- strict mode with the stall removed;
- one pending atomic per channel;
- while it is pending the channel takes no new packet; E1 keeps every other VC going, plus this channel's acks,
  completions and the next packet's route preload (reads only).

```cpp
// A flushed atomic waiting for its channel's earlier writes to be acknowledged. In the channel's state (memory), not
// in its progress: it is touched only when a packet carries a flushed atomic and while it waits.
struct PendingAtomic {
    uint64_t noc_addr = 0;     // copied at deferral, so the packet's slot retires as usual
    uint32_t value = 0;
    uint32_t proven_index = 0; // slots before it (from the channel's last proven point) had every write acknowledged
    bool pending = false;
};
template <size_t VC> struct ReceiverChannelState {
    ReceiverChannelProgress progress;
    SlotSourcesOf<VC> sources;
    [[no_unique_address]] Present<ReceiverChannel<VC>::delivers_flushed_atomics, PendingAtomic, VC> atomic;
};

// The step with both features:
//   [pending atomic? try to issue it]  ->  [handle the next packet if none pending]  ->  [preload the next route]
template <size_t VC, typename Connections>
FORCE_INLINE void step(...) {
    ... arrived, route_loaded, first-level acks (unchanged) ...
    bool may_take_packet = true;
    if constexpr (ReceiverChannel<VC>::delivers_flushed_atomics) {
        if (state.atomic.pending) {                                   // one load and one branch per pass
            may_take_packet = settle_pending_atomic<VC>(state.atomic, progress.next_index);   // out of line
        }
    }
    const bool handled_packet = may_take_packet && receive<VC>(...);  // a flushed atomic in the packet: deferred
    ... preload the next route, completions, store (§1.3) ...
}

// Out of line, run only while one is pending: the proof, then the atomic. Returns whether the channel may take a
// packet (no atomic left pending).
template <size_t VC>
__attribute__((noinline)) bool settle_pending_atomic(PendingAtomic& atomic, uint32_t next_index);
```

- **Deferral.** At the three flushed-atomic sites in `deliver`:
  - the standalone atomic increment with flush;
  - the fused write + atomic with flush;
  - a scatter whose last chunk is SEMINC_FLUSH.

  Each issues its writes as today, then copies `{noc_addr, value}` into `state.atomic` and sets `pending`, instead of
  `flush_write_to_noc_pipeline` and the increment. The slot retires and its completion goes back as usual (no held
  slot).
- **The proof is "acknowledged".**
  - `ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, id)`, i.e. NIU_MST_REQS_OUTSTANDING_ID(id) == 0.
  - It covers every ID of the slots from `proven_index` up to the deferred packet's slot. These are the completion
    bookkeeping's IDs (`CompletionTrids::of_slot`), at most NUM_TRANSACTION_IDS distinct ones, read oldest first.
  - An ID read at zero stays proven: the channel issues nothing while it waits, and its IDs are its own. The
    exception is ID 0 on VC0 while buffer 3's atomics and stateless doorbells carry tag 0 (Q5).
  - When all are proven: the atomic is issued (`noc_semaphore_inc` on LOCAL_NOC), `proven_index` becomes the slot
    after the deferred one, and `pending` is cleared.
  - Posted writes (forwards, posted inline writes) stay uncovered, as today.
- **Gate placement.** Outside the pass's opening read group, in the channel's step, after its acks.
  - The proof's NIU reads happen only while one is pending.
  - The gate itself is one L1 load and one branch per running channel per pass. Its cost is the first A/B (§5.1).
- **Drain.** A noinline `drain_pending_atomics(states)` spins on each pending proof and issues the atomic:
  - after `receiver::run`'s round loop, before `teardown()`;
  - on every exit of `single::run`'s inner and outer loops (router commands, termination).
  - The E1 park (context switch, PAUSE) stays open: Q6.
- **With the preload.** The packet that defers retires in that step, so the step's end preloads the next route (reads
  only). The passes that wait then do acks, completions and the gate. When the proof holds, the step issues the atomic
  and goes on to the next packet, whose route is already loaded.

### 1.5 Intermesh ingress (the landing)

- `fabric_set_2d_intermesh_landing_route` rewrites the packet's route buffer before the decode (out of line, once per
  packet on an ingress channel), so it moves to preload time: `Routing2D::load` lands, then loads.
- Safe: the packet is present (counted) and its slot is this router's until it is credited. The landing reads only
  that packet's header, and the later forward reads the rewritten buffer, as today.
- On the load-now path it lands then loads, as today.
- Cost: none moved per packet beyond the call's position. Static: the ingress configs are in bh_vc1.

### 1.6 Exit routers (the destination mesh)

- The draft reads `dst_start_mesh_id` (volatile) in `route_word`, at use. On an edge router its latency therefore
  stays at the decode.
- Option: preload it with the route as a third field. That is +1 carried value on edge routers only; the headroom check
  decides. Today's read order is destination mesh, then action bytes; with load-then-decode it becomes action bytes,
  then destination mesh. Q3.

### 1.7 1D

- The route is one word (`routing_fields.value`), as v4rx1r1 made it, and the preload carries one value.
- v4rx1rg's loss (+5.8 on rings, V4_RECEIVER_SHAPE §14) came from the old multi-field `LowLatencyRoutingFields`, spilled
  and reloaded twice.
- The draft: the preloaded word is first used at least 10 instructions after its load (18 on rings), at the next
  decode. It is not spilled; the first uses are the decode's `andi` (static).
- But the ring headroom fails (0/10, `uncredited` moves to the stack). The first-level-ack routers have no room for a
  carried value (§3.3), so the 1D line passes and the ring does not.

### 1.8 The data cache, teardown and loop exits

- `router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>()` runs at the start of `receive` and in the acks.
- The preload sits after this step's `receive`, so after an invalidate that follows the count read establishing the
  packet (§2). No extra invalidate is needed.
- Teardown and loop exits: the preload is reads only, so nothing to drain.
  - A one-ERISC router keeps its state across PAUSE and re-entry, and the preloaded route stays valid: the slot is not
    rewritten until it is credited.
  - E1's park leaves registers and memory as they are.

## 2. Ordering safety

**Claim.** A preload reads a slot only after a completed read of the channel's count has shown that slot's packet
present. So it reads that packet's bytes, never an earlier lap's.

1. **The link writes a packet's data before its count (the protocol; not re-verified here).** The far sender's Ethernet transfer
   lands the packet in this router's L1, then the packets-received stream register is incremented by the same ordered
   transfer. A read of the register that sees n packets therefore sees n slots whose data is in L1.
2. **A preload is issued only behind a completed count.** It runs at the step's end, on
   `progress.packets_ready != 0` after the packet was retired.
   - `packets_ready` was set from `arrived`, the value of this or an earlier pass's register read. That read's result
     was tested (in receive's refill, or in the acks' `arrived == 0`) before the step went on.
   - The preload's address and its issue are control-dependent on that test. On an in-order core a load after a taken
     branch on a loaded value is issued after the value returned.
   - The preloaded slot is the next one counted, so its data was in L1 before the count that included it returned.
3. **First-level acks.** The acks read each arrived packet's `src_ch_id` after testing `arrived` (control-dependent, as
   today) and add those packets to `packets_ready`. The preload's slot is counted by the same tested value, so the same
   argument holds. The acks don't change `next_index`; the preload follows the handled packet's advance.
4. **Why not in the pass-start read group.** The group issues every channel's count read and the TX status read back to
   back between two read-group boundaries (`begin_read_group` / `end_read_group`), and their values are tested after
   the group.
   - A header load placed in the group would be issued before the count's value is known: it is not dependent on it.
   - The L1 load can be served before the stream-register read returns, so a packet that lands between the two reads
     would be counted while its slot was read too early.
   - Hence: never in the group; only behind a tested count.
5. **The landing (§1.5)** writes only the present packet's route buffer, before any read of it, on this ERISC.
6. **One ERISC.** The same code runs on state in memory; the sender half doesn't touch receiver slots.

The flush-defer's proof is about acknowledgement of writes already issued, not about reading packets, so it needs no
count ordering. Its ordering is §1.4's strict rule: no write of a later packet is issued while an atomic waits.

## 3. Static numbers (offline drafts on v4rxg3; -O3 unless marked)

Drafts:
- **v4rxrp0:** codec copy only.
- **v4rxrp1:** byte form, preload with a second branch.
- **v4rxrp2:** byte form with the join.
- **v4rxrp3:** word form with the join, a draft-only cast.
- **Multi-downstream diagnostics:** v4rxg3ml / v4rxrp3ml, the v4bddml shape on the new base (every 2D router's
  downstream entries live).

All are pinned as v4rxg3 (non-fabric files, plus `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` in rp0-rp3).
Tools: `loopvars_maps/route_preload/` (`preload_use.py`, `linecmp.py`, the guard and headroom outputs).

### 3.1 Identity of the codec entry

v4rxrp0 against v4rxg3:
- -O3: 2,863 of 2,863 byte-identical.
- -Os: 2,831 identical. Of the 32 that differ, 12 only move code and 20 exchange two callee-saved registers (for
  example s4 <-> s5) with the same instructions in the same order. They are bh_2d_allinstr 8, bh_2d_codeprof 12,
  bh_torus 4, glx_torus 4, bh_vc1_allinstr 2 and bh_vc1_trim_rx0 2. Only VC1 builds use -Os in production.

### 3.2 Where the preloaded route is first used

`preload_use.py`: on every control-flow path from a route load, the instructions until the first reader of its
register.

| family | byte form (rp2): preload / load-now | word form (rp3): preload / load-now |
| --- | --- | --- |
| bh_2d, glx_2d (two ERISCs) | 1-2 / 1-2: `zext.b` right after the `lbu` | 9-10 up to the next pass's decode (`srl`) / 1+ |
| bh_2d_1erisc | 1-3 (`slli`/`or`) / 2 | 6+ (`sw`, the store after the completions) / 2-4 |
| bh_1d (word in both forms) | 10+ (`andi`, the decode) / 3 | the same |
| bh_ring | 18+ / 3 | the same |
| bh_torus | 1-2 / 1-2 | 1+ (some paths use it at once) |
| multi-downstream diagnostic (bh_2d, glx_2d) | - | 9-10+ / 1+ |

- The extension (test compile, `-mcpu=tt-bh -O3`): a volatile `uint8_t` read kept as a byte, or put into a struct of
  bytes or of 32-bit fields, gets `andi 0xff` / `zext.b` after the `lbu`.
- The row byte's extension lands right after the load. A volatile word read (`lw`) gets none. A non-volatile byte read
  gets none, but would leave the load's place to the compiler, against the codec rule.
- So with the byte form the 2D preload pays the load's latency at the preload. Today's load-now path pays it at the
  decode. The total is the same and nothing is hidden.

### 3.3 Carried values and the headroom check (`loopvars.py --headroom`, v4rxg3 -> draft: what the new field moves)

| family | byte form (rp2): pass / hot values moved / carried +n | word form (rp3) |
| --- | --- | --- |
| bh_1d | 10/10, none, +1 | 10/10, none, +1 |
| bh_2d | 28/28, none, +1..+5 | 24/28: deliver's NoC value and encoding on 4; +2..+6 |
| glx_2d | 48/48, none, +1..+5 | 44/48: deliver's value and address on 4; +0..+4 |
| bh_2d_1erisc | 28/28 | 28/28 |
| bh_2d_vc2 | 16/28: `uncredited`, deliver's value | 28/28 |
| bh_ring | 0/10: `uncredited` | 0/10: `uncredited` |
| bh_torus | 0/12: deliver's value, address, encoding | 4/12: deliver's value, address, `txq_status` |
| bh_torus_vc2 | 16/28: `txq_status`, the route byte | 16/28: the route words and their decode |
| bh_vc1 | 14/22: VC0's idle count, frame-due | 0/22: the route words and their decode |
| glx_express | 20/28: deliver's value, address | 20/28: `txq_status` |
| multi-downstream diagnostic, word form | - | bh_2d 24/28, glx_2d 44/48 (as above) |

The standard probe (headroom_g3.txt) already showed one extra carried value failing on rings (0/10) and tori (4/12),
and two on VC1 (10/22). The drafts match: the families with first-level acks and VC1 have no room for the field. On
the target, N/S mesh routers, it fits.

### 3.4 Loop size and stack accesses

Static instructions in the receiver loop, and stack loads + stores in it (ranges over the family's images):

| family, routers | g3 | byte form (rp2) | word form (rp3) | stack accesses g3 / rp2 / rp3 |
| --- | --- | --- | --- | --- |
| bh_1d | 488 | 515 | 515 | 4 / 12 / 12 |
| bh_ring | 447 | 464 | 464 | 9 / 17 / 17 |
| bh_2d E/W | 492-494 | 476-479 | 490-492 | 6-12 / 6 / 9 |
| bh_2d N/S | 488-596 | 493-622 | 536-606 | 6 / 11-15 / 16-20 |
| glx_2d E/W | 492-494 | 476-479 | 504-506 | 6-12 / 6 / 8 |
| glx_2d N/S | 488-689 | 493-705 | 515-693 | 4-6 / 11-15 / 9-23 |
| bh_2d_1erisc (receiver_pass) | 561-651 | 583-693 | 598-717 | 7-9 / 12-16 / 12-18 |
| bh_torus | 453-568 | 492-606 | 495-589 | 11-13 / 15-22 / 13-20 |
| bh_vc1 | 583-1317 | 638-1371 | 630-1364 | 6-42 / 11-57 / 16-64 |

Static size includes cold paths. In the N/S word form, about +23 of the growth is GCC duplicating the atomic
delivery path (`noc_semaphore_inc`'s line); the preload's own lines are +12 (two load sites) and +6 (byte extraction at
the decode).

### 3.5 Counted work per forwarded packet (what an estimate may use)

- Added, word form, on an N/S router:
  - the carried-count test (1);
  - the preload condition and address (about 4);
  - the stores after the completions (0 for the loop copy; 2 for a VC in memory or one ERISC);
  - byte extraction at the decode (2-4, replacing today's two `zext.b`);
  - about +4 to +8 in total.
- Removed: none. The loads move; they don't disappear.

## 4. Predictions (written before any run)

Rules:
- Counted work only; no stall-time estimate.
- Confidence marked.
- The earlier estimates on this core missed by 2-4x, so the ranges are wide on purpose.

Counted work says S1 is a cost of about +4 to +8 instructions per forwarded packet on E1, wherever E1 bounds the
row. A gain needs the overlap the static first-use numbers allow (word form on meshes). Its size is the A/B's to
find. V4_NS_FLOOR's PC-sample shares at the route word (7.9) and at the admission after it (17.2) on the
multi-downstream shape locate the stall; they don't size a gain (CLAUDE.md). **WRONG (§8.2): the admission's 17.2 is
not the route word's stall; preloading the route left it unchanged.**

| row (P150 unless marked) | bound side (from earlier A/Bs) | prediction for S1, word form, where applied | confidence |
| --- | --- | --- | --- |
| ew4352, hop3 (E/W write controls) | E0 (V4_NS_FLOOR §2: E/W 3-hop E0-bound; lead E1 ~118 against 144.7) | 0 (±0.5) | high |
| E/W scatter, atomic, fused (codec rows) | E1-sensitive (the codec cost showed there) | -2 to +3 | low |
| forwarded atomic / fused / scatter | E1-sensitive (+3.9 to +4.2 codec) | -3 to +3 | low |
| multicast scatter | E1-sensitive | -3 to +3 | low |
| N/S rows (ns4352, torus N/S writes) | not measured; the P150's N/S routers have one live downstream (V4_NS_FLOOR §2), so not the multi-downstream path | 0 (±0.5) | medium |
| multi-downstream diagnostic, 3-hop E/W | E1 (V4_NS_FLOOR: ml 174.5, E1-bound) | -15 to +5 | low |
| one ERISC E/W (ew4352, ewscatter, mcast) | one core | +1 to +4: the field in memory adds a store and a load | low |
| rings, tori (if applied) | E1 / first-level acks | +2 to +8: hot spills (`uncredited`, `txq_status`); cf. §8's B2 | medium (direction) |
| VC1 split (if applied) | - | +2 to +10: the route words spilled | medium (direction) |
| VC2 on (VC0 preloaded; VC2 delivers only) | - | 0 to +2 | low |
| latency (exp_latency.yaml, one packet at a time) | path | 0 to +3 cycles per hop (first packet after idle: load now, plus one test) | medium |
| Galaxy N/S rows (172.5-173.3, E1-bound) | E1 | -15 to +5 (as the diagnostic) | low |

S3 (flush-defer), counted:
- Rows that never flush: the gate adds 1 load and 1 branch per running channel per pass: 0 to +2.
- Single-stream flushed rows: about flat. The channel waits as long as today.
- Gains where something shares E1: other VCs, acks, one ERISC. Up to today's spin per signal (FLUSH_DEFER_DESIGN:
  220-290 cycles per signal on the v2 router, meas; v4's own unmeasured).
- Confidence: low.

## 5. Measurement plan (after the user's review; nothing queued)

### 5.1 P150, single-change A/Bs

Lead v4rxg3 against the stage. Same job, two passes in reversed router order, TT_FABRIC_NO_BW_TELEMETRY=1,
`row_ab_runs.sh`. Rows:
- the codec rows: forwarded atomic / fused / scatter, multicast scatter, E/W fused (p3_mixes / exp_2d_4352 via a
  census two1 subset, as jobs 427-428), and one ERISC E/W (ew4352, ewscatter, mcast);
- the E/W write controls: ew4352, hop3;
- rings (ring1, halfring), torus (torusns, torusns_scatter), VC1 (census_vc1_runs.sh), VC2 on (torusns,
  torusns_scatter, ew4352), one ERISC;
- the latency test, exp_latency.yaml (`suite_yaml lat`).

S3's gate-only arm (record + gate + drain, `deliver` unchanged) goes first, on rows that never flush. If it costs more
than 1-2 cycles per packet, rethink before the full S3.

### 5.2 The multi-downstream (N/S) path on the P150 first

- v4rxg3ml against v4rxrp3ml (the v4bddml shape on the new base, built offline here).
- Rows: the 3-hop E/W and turn rows of V4_NS_FLOOR §2 (sls_sl_runs.sh's set), two passes reversed.
- PC samples of the forwarding E1 (`bnd_pcs_runs.sh`): did the stall at the route word and the admission (7.9 + 17.2)
  move? Within one ERISC only (CLAUDE.md).

### 5.3 Then the Galaxy N/S rows

- Slurm, dry-run first.
- **Only on bh-glx-110-a\* nodes (`bh_sc36_6`, or the 110a pods); never bh_sc36_2 or other racks.**
- Lead against S1 on the N/S rows and the E/W controls, two passes reversed (the slot-study harness, as jobs
  167648 / 167673).

### 5.4 Each stage's offline gates before any device job

Full coverage at -O3 and -Os, guard_builds.py, the headroom check against headroom_g3.txt, size_report.py, and the swap
guard. The rings/tori/VC1 spills (§3.3) are expected to fail the headroom check; S1's scope (Q1) decides whether those
families get the field at all.

## 6. Stage plan

- **S1: route preload.** Word form, codec entry, the step of §1.3, on the channels Q1 decides. Then the measurement
  plan.
- **S2: the header-word arm (conditional).** Built only if S1's PC samples show the header word's load (payload size,
  send type, source) is next on the path. The same mechanism: one more carried word; the headroom check decides.
- **S3: strict flush-defer.** §1.4; gate-only A/B first (FLUSH_DEFER_DESIGN's test plan), then the full arm with the
  ordering stress test sketched there.
- **N/S fix 1 (partial presets on multi-downstream routers)** is queued after these and is independent
  (V4_RECEIVER_OWNERSHIP's planned section).

## 7. Open questions for the user

1. **Scope of the field.**
   - (a) Every forwarding channel: rings, tori and VC1 spill hot values (§3.3).
   - (b) Channels of routers without first-level acks: meshes, the N/S target. VC1 still fails with the word form
     (0/22).
   - (c) A per-channel feature the headroom check sets, e.g. `preloads_route`, off on rings / tori / VC1 for now.
   - The data favours (b)/(c). The feature-per-channel rule (readability anchor) would make it a feature of the
     channel's VC.
2. **The word view.** Reading the route buffer as words needs a word view in the header type (a union member in
   `HybridMeshPacketHeaderT`, a non-fabric file pinned in the variant), or the byte form, whose 2D preload hides
   nothing (§3.2).
   - Fewer carried values: the router's mesh coordinates as compile-time args (a host fact) would make the word
     indices and shifts constants: 1 carried value instead of 2-4 on N/S.
3. **Exit routers.** Preload the destination mesh too (+1 carried on edge routers), or read it at use (the draft)?
4. **The flush-defer gate's cost.** One load and one branch per running channel per pass on every row. Alternative: a
   bit kept with a carried value (no load, +0-1 register). The gate-only A/B decides.
5. **Proof IDs.** Keep VC0's buffer-3 tag on ID 0 (atomics and stateless doorbells, non-posted, would count against
   VC0's channel), or move it to a spare ID (FLUSH_DEFER_DESIGN step B)?
6. **E1 park.** Drain pending atomics before a context switch / PAUSE, or hold them across it? (Open since
   FLUSH_DEFER_DESIGN.)
7. **RELAXED flushes.** v4 treats them as strict, so they would take the deferred path. Fine?

## 8. Study A/B on the drafts (2026-11-02; jobs 457-459; P150; probe, not an adoption build)

Arms:
- **Multi-downstream proxy:** v4rxg3ml against v4rxrp3ml (word form; every 2D router's downstream entries live, as on
  V4_NS_FLOOR §2's v4bddml). Rows: `sls_sl_runs.sh` (base arm), the 2D and 1D sets.
- **Production shape:** v4rxg3 against v4rxrp3 (word form). Rows:
  - the census's 2D 4352 and mix suites (`rp_probe_runs.sh codec`);
  - one-ERISC ew4352 / ewscatter / mcast;
  - ew4352, hop3, ring1, torusns;
  - exp_latency.yaml (`rp_probe_runs.sh lat`).

Method:
- Same job for both arms of each row set, two passes in reversed router order, TT_FABRIC_NO_BW_TELEMETRY=1. Every test
  executed with 0 failures; the device was healthy after each job.
- Device time 27.3 min (369 + 840 + 429 s).
- Δ = draft - lead, mean of the passes, in cycles/packet (P150, 1.35 GHz).

**Correctness guard for the draft's word-view cast** (job 457, before anything else), v4rxrp3:
- mesh sanity 96/96;
- inline write then flushed atomic: 1D 2/2, 2D 2/2; main's control 2/2, 2/2.

No failure and no hang.

### 8.1 Multi-downstream proxy (v4rxg3ml / v4rxrp3ml; passes a, b)

| row | lead a / b | preload a / b | Δ |
| --- | --- | --- | ---: |
| 3-hop E/W write 1024 / 4352 | 178.3 / 178.3; 179.8 / 179.8 | 180.7 / 180.7; 181.5 / 181.5 | +2.4 / +1.7 |
| turn (SlsTurn4) 1024 / 4352 | 171.0 / 171.0; 171.7 / 171.7 | 174.1 / 174.1; 176.3 / 176.3 | +3.1 / +4.6 |
| 1-hop N/S write 1024 / 4352 | 117.1 / 117.1; 127.3 / 127.3 | 119.0 / 119.0; 129.0 / 129.0 | +1.9 / +1.7 |
| 1-hop E/W, E/W bidir, multicast, row merge (2D) | | | +0.2 to +0.5 |
| 1D line 1-hop / 3-hop / row merge, 1024 | 95.4 / 118.2 / 108.8 | 98.5 / 119.2 / 111.0 | +3.0 / +1.0 / +2.2 |
| 1D, 4352 | 124.5-125.8 | the same | -0.1 to 0 |

### 8.2 PC samples, the proxy's 3-hop E/W row

Setup: `bnd_pcs_runs.sh`, row hop3, the forwarding E1 at chip 2 / channel 4, a West-facing router with two
downstreams. One sampled run per arm: lead 179.66, preload 181.41. Cycles/packet = share x rate, within one ERISC only
(CLAUDE.md).

| lines (inline chains) | lead | preload |
| --- | ---: | ---: |
| route: the codec's load and decode at use (lead: one line, the X byte) | 12.0 | 7.9 (preload in step 2.5, load-now in receive 1.0, byte extraction at the decode 4.4) |
| admission: `any_without_free_slot` | 16.6 | 18.4 |
| admission: `admit`'s re-read group | 6.1 | 7.1 |
| header word | 1.7 | 1.8 |
| command-buffer writes (`NOC_CMD_BUF_WRITE_REG`) | 20.3 | 22.1 |

- The route word's samples fell by ~4.1. The admission's did not: +1.8 and +1.0.
- The admission samples are spread over its loop's instructions (1.1-2.0 cycles/packet per address, no single
  first-use stall), so they read as executed work, not a load's wait.
- V4_NS_FLOOR §2's hypothesis that the route word's header load latency lands on the admission test is contradicted
  by this single-change A/B: the route word arrives early here and the admission costs the same or more. Marked WRONG
  there.
- What the admission's ~17 is remains a hypothesis: the multi-downstream bookkeeping's own instructions (the per-entry
  free-slot tests).

### 8.3 Production shape (v4rxg3 / v4rxrp3; passes a, b)

| row | lead a / b | preload a / b | Δ | §4 prediction | verdict |
| --- | --- | --- | ---: | --- | --- |
| ew4352 | 127.0 / 127.0 | 127.1 / 127.1 | +0.1 | 0 ±0.5 (high) | holds |
| hop3 | 146.7 / 146.7 | 146.4 / 146.4 | -0.3 | 0 ±0.5 (high) | holds |
| E/W scatter (4352 suite; mix) | 142.4 / 142.3 | 146.0 / 146.0 | +3.6 | -2 to +3 (low) | **WRONG** (beyond the range) |
| E/W bidir scatter | 143.7 / 143.7 | 147.8 / 147.8 | +4.1 | (as E/W scatter) | **WRONG** |
| E/W atomic | 130.1 / 130.1 | 133.4 / 133.3 | +3.3 | -2 to +3 | **WRONG** |
| E/W fused | 150.1 / 150.1 | 152.6 / 152.6 | +2.6 | -2 to +3 | holds (no gain) |
| forwarded atomic / fused / scatter | 136.5 / 156.8 / 148.7 | 137.2 / 157.5 / 149.8 | +0.8 / +0.8 / +1.0 | -3 to +3 | holds (no gain) |
| multicast scatter / write | 173.1 / 141.7 | 173.3 / 141.9 | +0.2 / +0.2 | -3 to +3 | holds |
| 3-hop E/W scatter / write (4352 suite) | 150.2 / 146.8 | 150.7 / 146.5 | +0.4 / -0.3 | | |
| N/S 1-hop write (4352 suite) | 127.3 / 127.3 | 129.1 / 129.1 | +1.8 | 0 ±0.5 (medium) | **WRONG** |
| N/S scatter / atomic / fused (4352, mix) | 145.0 / 132.4 / 152.4 | 158.2 / 140.4 / 160.8 | +13.2 / +8.0 / +8.4 | (N/S rows: 0 ±0.5) | **WRONG** |
| N/S mix writes (16 B .. 4352, chunks) | 116.4-129.9 | 118.1-131.8 | +1.4 to +4.3 | 0 ±0.5 | **WRONG** |
| one ERISC: ew4352 / ewscatter / mcast | 200.0 / 242.5 / 331.6 | 212.9 / 257.8 / 344.0 | +12.9 / +15.3 / +12.4 | +1 to +4 (low) | **WRONG** (3-4x) |
| ring1 | 140.8 / 140.9 | 145.6 / 145.6 | +4.7 | +2 to +8 (medium, direction) | holds |
| torusns | 149.2 / 149.2 | 164.0 / 164.0 | +14.8 | +2 to +8 | **WRONG** (size; direction right) |

The one-ERISC rows are the same to 0.1 in both passes.

Latency (median net round trip, ns; 1000 samples per case; passes a / b):

| case | lead | preload | Δ |
| --- | --- | --- | ---: |
| 1-hop E/W write 16 / 4096 | 1348 / 1348; 1692 / 1687 | 1348 / 1349; 1693 / 1702 | 0 / +8 |
| 1-hop E/W fused 16 / 4096 | 1407 / 1411; 1703 / 1705 | 1413 / 1408; 1704 / 1695 | +2 / -4 |
| 1-hop N/S write 16 / 4096 | 1336 / 1339; 1681 / 1691 | 1347 / 1348; 1692 / 1693 | +10 / +6 |
| 1-hop N/S fused 16 / 4096 | 1402 / 1403; 1685 / 1686 | 1413 / 1408; 1704 / 1700 | +8 / +17 |
| 3-hop E/W write 16 / 4096 | 3322 / 3320; 4243 / 4244 | 3306 / 3307; 4248 / 4244 | -15 / +3 |
| 3-hop E/W fused 16 / 4096 | 3380 / 3370; 4259 / 4258 | 3378 / 3379; 4254 / 4259 | +3 / -2 |

- §4's "0 to +3 cycles per hop" holds on E/W, within the pass-to-pass spread (up to ~10 ns).
- **WRONG on N/S:** +6 to +17 ns per round trip, about 4-11 cycles per hop, in both passes.
- The 3-hop 16 B write is 15 ns faster in both passes; the cause is not known.

### 8.4 What this says (hypotheses where marked)

- **The preload, as drafted, pays nowhere in these rows.**
  - Every throughput row is flat or costs. The cost is +1 to +5 on E/W small-packet rows, +8 to +13 on N/S delivery of
    scatter, atomic and fused, +13 to +15 on one ERISC, and +15 on the Y-torus N/S row.
  - On the multi-downstream proxy, the shape the fix targets, it costs +1.7 to +4.6.
- **Where the costs come from, hypothesis (not isolated).**
  - N/S routers carry two route words and two byte shifts (the word form). The static headroom check already moved
    deliver's NoC value and scatter encoding to the stack on 4 of 28 bh_2d configs (§3.3); the N/S delivery rows are
    where that shows.
  - On one ERISC the field lives in the in-memory state, with a load and a store each packet, and receiver_pass grew
    +37 to +66 instructions (§3.4).
  - Rings and tori failed the headroom check (§3.3).
- **What the gain hypothesis assumed and the A/B contradicts.**
  - The route word's latency was part of the N/S floor's admission cost (V4_NS_FLOOR §2). With it preloaded, the route
    samples fell ~4 cycles/packet but the admission's did not move. The A/B rules out the route word as the cause.
  - So the ~17 the fix aimed at is not the route word's latency (§8.2). §4's reading of the 7.9 + 17.2 as the stall
    the preload addresses is **WRONG** (marked there): at most the route part (~4 here) is removable, and the added
    work and register pressure cost more than that.
- Whether to build S1 at all, its scope (§7 Q1) and the word-view question (Q2) are the user's decisions. Nothing
  further is built or run.

Files: results `slsprl{a,b}sl_*`, `bndp_prl{g3ml,rp3ml}_*`, `cenprl{a,b}_*`, `rabprl{a,b,1a,1b}_*`, `prllat{a,b}_*`,
`prlv_v4rxrp3_mesh`, `rxoi_*`. Script `consolidation/rp_probe_runs.sh`. Sample splits by chain category: the
dumps from `pcs_lines.py --dump`, categorised by function, in this section.
