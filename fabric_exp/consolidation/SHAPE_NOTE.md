# Router shape note (draft for review, 2026-10-26)

A proposal for the shape of the Blackhole fabric router kernel: how its code is organized, where its state lives, and
which of the study's optimizations it keeps. The goal is a router that is fast **and** that the people who maintain
the fabric can read and change without perf swinging. Every choice below cites the measurement behind it (RESULTS
§n, LEDGER row, SHAPE_INVENTORY.md, ROUTER_MAP_V3.md). Code blocks are sketches of structure, not compilable code.

Nothing here changes the packet format, the credit protocol, the connection protocol or the fabric APIs workers use.
The changes are the router kernel's internal structure, a few host-side compile-time arguments, and small NoC API
additions.

## 1. Why a new loop shape, and not more patches to main's

Main's router runs one loop for both ERISCs, inlined into `kernel_main`: sender steps and receiver steps interleaved
per channel, gated by compile-time "serviced" masks, with channel state in objects that live in `kernel_main`'s frame.
It is readable. It is also slow (222 cycles/packet on a 1-hop write where the wire allows 125 at 400G) and, more
importantly, it is a shape where local changes do not behave locally:

| Change tried (on main's loop shape, or a v3 loop that still had it) | Result | Ref |
| --- | --- | --- |
| A `[[likely]]` hint on one branch | 1-hop writes 132.7 → 138.3 | §49 (bit 25: §40) |
| Forcing one helper inline | E/W rows +4 to +10 | §40 |
| A smaller sender-channel struct (less state) | small writes +10, 4 KiB writes +15 | §32, §77, §78 |
| Plain volatile loads instead of the asm load groups | small writes +7.5 | §46 |
| The main loop `noinline` on both ERISCs | receiver +18 to +108 instructions, scatter +6 to +10 | §78 |
| One shared forwarding body instead of per-channel copies | forwarding +20 to +34 | §35, §36 |
| Six semantically null cleanups in the consolidated (v3a) loop | receiver scatter +4 to +6 | §94 (P3) |
| One more check in per-round code that sits inside the loop, as main's does | the whole loop re-laid out (600–1,600 differing lines) | §97 (P4, A1) |
| Building at -Os (production VC1 builds do) | +19 to +28 cycles/packet | §8, §82 (P1) |

The study's patch phase (§17–§81) got 1-hop writes from 222 to 132, but each step needed hardware to find out what an
edit did; eight ideas were lost to the code around them and five more traded one row against another (LEDGER part 1,
LOST-CONTEXT and MIXED). The cause is
structural: one loop serving both roles, at the limit of the register file, inlined into a 3,000-line function, so
the compiler's allocation and layout of the whole loop move with any edit.

v3 (this study's consolidation) separated the roles and put each in its own small loop function. That part worked:

- Edits that change nothing semantically move suite means by ±0.4 at most (jitter band, T14), and the loop's code
  does not change under cold edits (P4).
- -Os costs 70% less than in the consolidated build that kept main's loop shape (v3a; §93, §94).
- On a Galaxy (200G) it runs 1.73–1.85x main on 2D 1-hop writes from 16 B to 2 KiB (1.2–1.6x on scatter, fused,
  multi-hop and multicast rows) and saturates the link from ~2.3 KiB (RESULTS §98, Galaxy).

What did not work is the rest of v3's shape: it was grown by consolidating and patching, under a byte-identity
constraint, so it kept main's loop as a fallback (three programs, ~6,600 lines), hard-codes four channels, puts every
receiver feature in one 245-line function, and touches NoC registers directly. This note keeps v3's structural
result and redraws the rest.

## 2. Requirements

1. **Every production config, one program per role.** Section 1 of SHAPE_INVENTORY.md: 1D and 2D, rings and tori with
   deadlock avoidance, express (5 VC0 sender channels on every router, Z-facing routers), multi-mesh VC1, VC2, the
   mux, UDM, first-level ack, telemetry and profiling, single-ERISC (Wormhole, and Blackhole with two-ERISC mode off).
   Today v3's own programs cover Blackhole 2D/1D VC0 and VC1 (and the sender side of rings and tori); everything
   else falls back to main's loop.
2. **Channel counts are compile-time lists, never fixed.** No `4`, no `raw[4]`, no `index_sequence<4>`.
3. **Performance:** within a few cycles/packet of v3q on suite means at -O3 and -Os, no row worse by more than 3; on
   configs v3q does not cover, not slower than main. Robust over peak (R1–R3): no timing cliffs.
4. **Stability:** ordinary edits stay inside the jitter band (±0.5 on means); perturbation sweeps stay smooth.
5. **Readability:** domain names (sender, receiver, channel, downstream, completion); one concept per function; hot
   functions short enough to read in one screen; each optimization carries one line saying what it saves and where
   that was measured.
6. **Rules carried over** (user decisions in the study): no receiver work on the sender ERISC; one worker channel per
   router; the worker connection is checked on every pass; NoC programming only through the NoC APIs, behind named
   helpers; no branch-layout hints, no register pinning, no source forms that rely on a particular compiler behavior.

## 3. Principles

Each principle, the evidence for it, and what it replaces.

**P-A. Each role is a loop function of its own; per-round and rare work is out of line.**
The function boundary is what keeps a loop's code independent of everything else (P4, A1, E1-11: cold edits never
moved the loop; inlined, three of four did). The loop holds only per-pass code; termination checks, heartbeat,
context switch, connection open/close and telemetry flushes are small `noinline` functions called once per round or
on an event.

**P-B. State placement is a data-layout decision, made explicitly.** Four kinds of state, four places:

| Kind | Examples | Where it lives |
| --- | --- | --- |
| Build-time constant | slot counts and ring bounds, stream ids, buffer bases, semaphore and registration addresses, counter addresses, command-buffer ids | `constexpr`, from named compile-time args |
| Set once at init, per router | downstream NoC coordinates, mesh coordinates | command-buffer state (programmed once through the stateful NoC APIs) or a loop-function local |
| Per connection | the connected worker's semaphore NoC address, a forwarding channel's upstream credit address | a small table, read on use |
| Per pass or per packet | slot pointers, counts, the in-flight mask, the far ring cursor, receiver channel state | plain locals of the loop function |

Why: the inventory shows where today's cycles go. The receiver keeps its per-channel state as loop locals and spends
0.0–0.2 cycles/packet on memory traffic; its only cost is the downstream cursors it reaches through a pointer (5.6–7.4
on forwarding routers). The sender keeps its per-pass values inside a large struct (`ForwardingState`: the in-flight
mask and an array of channel records) and spends 8–12 cycles/packet on 1-hop rows and 16–21 on forwarders loading
and storing them. Locals are register candidates by construction; members of a large aggregate usually are not. This
is plain data layout, not coaxing the compiler: nothing depends on a particular allocation, and where the register
file is full (every loop today uses all 13 callee-saved registers) the per-visit records stay in an explicit table,
read once and written once per visit.

The runtime-arg diff (SHAPE_INVENTORY §2) shows that every build-time constant above is already the same within each
router binary, so making them compile-time adds no binaries. Only the downstream NoC x coordinate varies, and it stays
runtime (compile-time coordinates would add 14–63% binaries on Galaxy 2D configs).

**P-C. Features are compile-time components with their own small state types.** Routing (1D, 2D, express), atomic
ordering (strict, deferred), completion batching, first-level ack, bubble flow control, intermesh landing and exit,
the UDM relay, telemetry: each is a type chosen from the config, with a no-op form when off. A component's state is a
small value type with named operations (one word underneath where registers are tight), never an anonymous bit-packed
`uint32_t` decoded inline.

**P-D. NoC and Ethernet hardware only through APIs.** Stateful NoC APIs for presets (local writes, forwards, credits),
the TX-queue helpers and the link-transport policy (`SenderLinkTransport`, `ReceiverLinkTransport`, main #61974) for
Ethernet. Where an API is missing, add it (section 6), never `NOC_STATUS_READ_REG` or `NOC_CMD_BUF_WRITE_REG` in router
code.

**P-E. Keep the measured optimizations, in the form that was measured; drop the ones worth ≤ ~2 cycles that cost
readability.** Section 5 lists what carries over.

## 4. Structure

### Files

| File | Holds |
| --- | --- |
| `fabric_erisc_router.cpp` | `kernel_main`: one init and teardown skeleton for every role (the ERISC syncs in one place, review must-fix C3), then the role's loop |
| `router_config.hpp` | the compile-time configuration: channel lists per VC, per-channel and per-downstream traits, feature selection |
| `router_sender.hpp` | the sender program: worker channel, forwarding channels, far ring, the pass |
| `router_receiver.hpp` | the receiver program: the receive step as stages, the pass |
| `router_routing.hpp` | routing policies: 1D, 2D, 2D express; intermesh landing and exit |
| `router_completions.hpp` | completion tracking and credit batching (holds bounded by the ring, the bounded wait) |
| `router_atomics.hpp` | atomic ordering: strict, deferred (the relaxed-flush state machine with a named state type) |
| `router_downstream.hpp` | downstream connections: credits, forwards, setup |

Single-ERISC configs compose the same sender and receiver passes in one loop (section 4.4).

### 4.1 Compile-time configuration

```cpp
// router_config.hpp (sketch)
template <size_t ID> struct SenderChannel {            // one per sender channel of this router
    static constexpr ChannelKind kind = ...;          // Worker, Forwarding, (VC2) Worker
    static constexpr Vc vc = ...;
    static constexpr uint32_t num_slots = ..., first_slot = ..., last_slot = ...;
    static constexpr uint32_t free_slots_stream = ..., completion_counter = ...;
    static constexpr bool bubble_flow_control = ...;  // injection channel on a deadlock-avoiding ring
};
template <size_t ID> struct ReceiverChannel { ... };   // ring bounds, packets-received stream, trid range, ...
template <Vc VC, size_t D> struct Downstream { ... };  // ring bounds, credit streams, command buffers

using Vc0Forwarding = ChannelList<1, 2, 3>;            // from the config: <1> in 1D, <1, 2, 3, 4> with Z
using Routing = select_routing<config>;                // Routing1D, Routing2D, Routing2DExpress
using Atomics = select_atomic_ordering<config>;        // StrictAtomics, DeferredAtomics
```

All values come from named compile-time args the host already computes (the addresses that are runtime args today
move over; SHAPE_INVENTORY §2). `ChannelList` replaces every fixed `4`.

### 4.2 Sender program

```cpp
// router_sender.hpp (sketch)
struct WorkerChannel {                // loop locals, one object
    uint32_t slot, sent, completed;   // + acked with first-level ack
    uint32_t expected_connection;     // compared with the connection semaphore every pass
};
struct FarRing { uint32_t free_slots, next_slot; };     // the far receiver's ring, one per VC
struct ForwardingRecord { uint32_t slot, sent, completed; };  // per forwarding channel, in a table

__attribute__((noinline)) void run_sender(ConnectionTable& connections) {
    WorkerChannel worker = ...;
    FarRing far_vc0 = ..., far_vc1 = ...;
    uint32_t in_flight = 0;                              // forwarding channels with packets not yet completed
    ForwardingRecord records[Vc0Forwarding::size + Vc1Forwarding::size];
    while (next_round()) {                               // out of line: termination, commands, heartbeat
        for (uint32_t pass = 0; pass < passes_per_round; pass++) {
            const FreeSlots free = read_free_slots<Vc0Channels>();     // one grouped read per VC
            visit_worker(worker, free, far_vc0, connections);          // send, credit, connection check
            if (has_forwarding_work<Vc0Forwarding>(free, in_flight)) {  // one test when nothing is forwarded
                for_each<Vc0Forwarding>([&](auto ch) { visit_forwarding(ch, free, far_vc0, records, in_flight); });
            }
            ... VC1, VC2 the same way
        }
        end_round(...);                                  // out of line
    }
}
```

- The worker visit is a function of its own (today ~50 lines inline in the loop), with the connection check in it,
  every pass (rule).
- The per-pass values (the far ring cursor and the in-flight mask) are loop locals. Today they cost 9–11 cycles/packet
  on every row as stack loads (SHAPE_INVENTORY §3).
- A forwarding channel's record is read and written once per visit from an explicit table. On forwarding routers that
  is ~7.5 cycles/packet today; whether a slimmer record (slot and outstanding count) does better is to be measured
  (LEDGER E0-14 saw ±5–10 swings in an earlier, less insulated loop).
- The grouped free-slots read generalizes to the channel list. With Z there are five VC0 channels; a fifth read in one
  group made it cost ~14 in §17 (four cost ~9.5 today), so the group splits into four plus one where five exist (to be measured).
- No calls on the pass path (E0-11: forwarded 16 B 119.4 → 112.3). Upstream credit writes and the TX-queue wait inline.

### 4.3 Receiver program

```cpp
// router_receiver.hpp (sketch)
template <typename Ch>
FORCE_INLINE bool receive_step(Ch& ch, Downstreams<Ch>& ds, Completions<Ch>& completions, Atomics& atomics) {
    if (!ch.has_packet()) {                               // poll: packets-received count as a lower bound (E1-1)
        completions.idle_pass(ch);                        // partial batches after idle passes (E1-28)
        return false;
    }
    const PacketView packet = ch.front();                 // header words, read once
    const Route route = Routing::decode<Ch>(packet);      // deliver here? which downstreams?
    if (!ds.admit(route)) {                               // cached downstream credits (E1-2)
        completions.idle_pass(ch);
        return false;
    }
    const Trid trid = ch.next_trid();
    atomics.before_issue(packet, route, trid);            // deferred-atomic ordering; no-op when strict
    ds.forward(route, packet, trid);                      // 0..N downstreams, preset command buffers
    if (route.delivers_here()) {
        deliver(packet, trid, atomics);                   // write / scatter / atomic / fused arms (E1-10, E1-17)
    }
    ch.retire(trid, route);                               // slot record, packets-received update
    completions.after_packet(ch);                         // batched credits to the far sender (E1-5, E1-25, E1-28)
    return true;
}

__attribute__((noinline)) void run_receiver(...) {
    ReceiverState<0> rx0 = ...;                           // loop locals, as today
    DownstreamCursors<0> ds0 = ...;                       // loop-carried slot and credit count as locals
    while (!termination_requested()) {                    // out of line
        for (uint32_t pass = 0; pass < passes_per_round; pass++) {
            receive_step(rx0, ds0, completions0, atomics);
            ... VC1, VC2 the same way
        }
        end_round(...);                                   // out of line
    }
}
```

- Today's `rx_channel_step` (245 lines) becomes the stages above, each a named function with one job. 1D and 2D
  differ only in `Routing::decode` and the forward fan-out, not by `#if` inside one body.
- Intermesh landing and exit are part of the 2D routing policy (they decide the route), not a branch in the step.
- The deferred-atomic resolution (today ~60 lines inline, three raw NIU status reads, state packed as
  `0x100 | b << 4 | a`) moves into `DeferredAtomics` with a named state type and the NoC API accessors of section 6.
- Downstream cursors: the per-forward values (slot pointer, cached credits) become loop locals of the receiver loop,
  the ring bounds become constants (today 5.6–7.4 cycles/packet on forwarders through a pointer, ~1.3 of it the
  constant bounds). With up to four downstreams that is eight more locals; if they do not fit, the cursor table stays
  in memory and only the constants move (to be measured).
- Rings, tori and first-level ack run this program too (today they fall back; half-ring multicast at -Os is +14,
  LEDGER E1-31). Completion batching under deadlock avoidance follows the same ring-bounded rule (R2), with the
  first-level ack as its own component.

### 4.4 Single-ERISC configs

Wormhole (one ERISC per Ethernet core) and Blackhole with two-ERISC mode off run both roles on one ERISC:

```cpp
__attribute__((noinline)) void run_both(...) {
    ... the sender's and the receiver's locals
    while (next_round()) {
        for (uint32_t pass = 0; pass < passes_per_round; pass++) {
            sender_pass(...);      // the same functions the two-ERISC programs use
            receiver_pass(...);
        }
        end_round(...);
    }
}
```

This is main's arrangement, built from the new components. Its register pressure is the sum of both roles; it is
held to "not slower than main" and measured offline first (no Wormhole hardware in this study).

### 4.5 Init and teardown

One `kernel_main` skeleton for every role: parse args, set up this role's side, the ERISC syncs (named phases, a
mismatch detectable instead of the current comment-only contract, review C3), the handshake, then the role's loop and
teardown. The receiver-only and sender-only init split (E0-6: the sender's `kernel_main` 6,416 → 4,520 B, deepest
stack 1,408 → 800 B) stays. Downstream setup goes through the adapter's own init functions instead of a hand-written
mirror of them (review a2).

## 5. What carries over

**From v3 (measured, kept):**

| Item | Value | Ref |
| --- | --- | --- |
| Own `noinline` loop per role, per-round work out of line | stability; receiver means −3.8 to −8.4 vs v3a at -O3 | P4, A1, E1-11 |
| Packets-received count kept as a lower bound; running slot pointer | basis of the receiver program | E1-1 |
| Cached downstream credits (reload at zero) | forwarding/multicast −4 to −8 | E1-2 |
| Local-write preset through the NoC API | scatter −6 | E1-17 |
| Direct write / scatter / atomic / fused arms | scatter −7 to −12, atomic −4 to −11 | E1-8, E1-10 |
| Exact completion check from per-slot records | fixes v3e1's local-delivery cost | E1-5 |
| Completion batching with ring-bounded holds | VC1 −12 to −83; no bursty losses | E1-28, E1-35 |
| Bounded wait for a batch's newest write before crediting | means −1.6 to −4.8 | E1-24, E1-25 |
| Decode fixes (both route bytes read at once); 1D decides once | N/S regressions gone; level-independent forwards | E1-14, E1-19 |
| VC1 admission on VC1's own credits | correctness (main is correct; v2/v3a were not) | E1-4 |
| No calls on the sender's pass path | forwarded 16 B 119.4 → 112.3 | E0-11 |
| Worker connection check as one load and compare; open/close out of line | per-pass rule kept, ~3 instructions | E0-9 |
| Worker credit = completed count | one stack load and store less per credit | E0-8 |
| Exact idle predicates (masked compare; bit test only where exact) | a single-bit test hung VC1 (2-slot over-fill) | E0-10 |
| Grouped stream-register reads | ~9.5 cycles for four registers | P7, ROUTER_MAP_V3 |
| Keepalive 1000 as a host setting on Blackhole | 400G: removes the 187.6 floor; 200G: +5 to +7 without it | H2, §98 Galaxy |

**From v3, dropped:** the generic loop as a fallback; mirroring the generic loop's predicates
(`generic_loop_visits_on_vc0`, `tested_alike`); the fixed group of four; anonymous bit-packed state; raw NoC
register access; `optimize("Os")` on cold functions (P10); fences that order nothing (P8, measure without them).

**From main, kept as is:** the packet header and routing fields, the credit and connection protocols, the worker and
router-to-router adapters' interfaces, the link-transport policy (#61974), telemetry and profiling formats, the host
builder's structure.

## 6. Host and API changes

| Change | Why | Size |
| --- | --- | --- |
| Pass the router's fixed addresses (semaphores, registration, location-info, buffer bases, ring bounds) as named compile-time args | constants in the kernel at no extra binaries (SHAPE_INVENTORY §2) | builder: move values from `get_runtime_args` to the named-arg map |
| Worker connection semaphore address as a compile-time arg | removes the pointer load on every pass | LEDGER E0-16 |
| VC1 builds at -O3 | v3 fits (v3z: 25,096 of 26,624 B); -Os was for v2's size | `get_kernel_opt_level`, P12 |
| NoC API: outstanding writes per transaction id (count, not only "sent"/"flushed") and the NIU write-ack counter | the completion check and the deferred-atomic resolution without raw status reads | LEDGER E1-20 |
| NoC API: whatever the forward and inline-credit paths still program by hand in `fabric_edm_packet_transmission.hpp` (15 `NOC_CMD_BUF_WRITE_REG` today) | P-D | to be listed when ported |
| `NocFlushType` in the atomic headers (RELAXED flushes) | the deferred-atomic optimization needs it | its own PR (review C5) |

## 7. Validation

- **Offline, every change:** builds of every coverage config (1,435 ELFs: Blackhole and Wormhole, every topology
  including Galaxy express, every feature) at -O3, -O2 and -Os; program mix (`stack_scan.py`): no config on a
  fallback; loop stack traffic and size against v3q; ordinary-edit check (`jitter_edits.py`, `fncmp.py`).
- **P150x8 (broker):** the study's suites (4352, mix, flushed mix, 1D, VC1) at -O3 and -Os against v3q and main;
  bursty-sender and perturbation sweeps (R3); PC-sampled cost map (ROUTER_MAP_V3 method) for the new loops; validation
  yamls (data-checked) for every topology.
- **Galaxy (Slurm):** mesh, linear, ring, torus suites and the size sweep against v3q and main; express once a Galaxy
  with express cabling (or a matching descriptor on a standard Galaxy) is available.
- **Acceptance:** section 2's numbers; any loss traced to a cause before it is accepted.

## 8. Plan

1. **New files beside main's router** (user decision): a new kernel source selected by the kernel-version switch, plus
   the headers of section 4. Main's `fabric_erisc_router.cpp` and v3 stay untouched. Init and teardown are adapted from
   main's `kernel_main` (the tested path), restructured into the one skeleton of 4.5; the loops are new.
2. **Bring-up, one config family at a time,** each passing the data-checked validation yamls before the next:
   Blackhole 2D (two ERISCs) → 1D → rings and tori (deadlock avoidance, first-level ack) → Galaxy express
   (`express_links_8x4_mesh_graph_descriptor.textproto`, the Galaxies' ex4 links) → multi-mesh VC1 → VC2 → the mux and
   UDM → single-ERISC on Blackhole → Wormhole (offline builds here; hardware runs on the other cluster by the user).
   Atomics strict only (no deferral) until the shape is reliable.
3. **Measure against v3q and main** after 2D and after each family: suites at -O3 and -Os, cost map per ERISC (PC
   sampling), jitter band.
4. **Host compile-time args** and the NoC API additions; the constants move into both programs.
5. **Galaxy runs** (torus, express, the size sweep) and the write-up for the team.

Each step keeps the previous one's numbers as its baseline and is reviewed before the next.

## 9. Decisions (user, 2026-10-26)

1. **Wormhole and single-ERISC are in scope.** Wormhole is built and checked offline here; its hardware runs happen on
   another cluster, outside these hosts.
2. **New files** first, beside main's router; see how it evolves.
3. **Express is testable on the Galaxies via Slurm:** each is a 2D torus with an ex4 express link; use the matching
   mesh graph descriptor.
4. **Deferred atomics off** for now: bring up one reliable router shape, then try to fuse them in.
