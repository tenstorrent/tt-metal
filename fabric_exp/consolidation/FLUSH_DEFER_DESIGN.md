# Parked design: strict-flush atomics that do not block the receiver

Status: design and audit done (2026-11-01, a design agent, read-only; nothing built). Parked for the feature pass after
the receiver beautification pass. Decisions below marked (user) were agreed in chat; the rest is the agent's proposal,
not yet reviewed. Labels: code = read from code, meas = measured, hyp = hypothesis.

## Today

A flushed atomic (standalone atomic increment with flush, fused write + atomic with flush, scatter whose last chunk is
SEMINC_FLUSH) calls `flush_write_to_noc_pipeline(channel)` in `deliver` (router_local_delivery.hpp `local_atomic_inc`),
which spins until the earlier writes are acknowledged, then issues the atomic: every ID of the channel's range with
deadlock avoidance; the global ERISC-wide ack counter (`ncrisc_noc_nonposted_writes_flushed`) on Blackhole without it.
The spin blocks everything on E1 (first-level acks, completions, credit frames, other VCs' channels) and, on one-ERISC
routers, the sender half. ~~Main's router: flushed fused ~495 vs ~154 cycles/packet unflushed (meas, RESULTS §63).~~ **WRONG (2026-11-02, report data check): those are kernel v1 at the study flags, not main as shipped; neither main as shipped nor v4 has been measured with flushing. Study builds: 522.8 vs 152.0 on E/W fused 4352 B (RESULTS §58).**
History: RESULTS §63-§64 (main's deferral staged in write_at_cmd_buf: -55 on flushed fused, +2 to +3 on every E1-bound
row, likely code growth from three inlined drain sites, hyp), V4_FEATURE_SPEC.md §14.5.

## The idea (user) and decisions (user, 2026-11-01)

When a packet needs a strict-flush atomic, issue its writes, note that the channel has a pending atomic, leave the
handling step, and continue everything else as today. When the router reaches that channel again, before any decoding
or handling, check whether the earlier writes have landed: if so issue the atomic and proceed, else move on. Other VCs
are not blocked; acks, completions and credit frames keep going.
1. The proof is per channel (its transaction-ID range, writes acknowledged), not the global counter (other VCs'
   writes would starve it; a snapshot of a global counter is unsound since acks arrive out of order), with a fallback
   for writes the channel issues without its IDs.
2. The atomic's NoC address and value are copied into the channel's state at deferral, so the slot retires and its
   completion goes back as usual (no held slot: held credits under bubble flow control collapsed rings, RESULTS §98).
3. While the atomic is pending the channel takes no new packet (no local writes, no forwards); its acks and
   completions continue.
4. Fast path: check the proof once when the atomic comes up; issue at once if the writes have landed.
5. Drain pending atomics at teardown and wherever the loop exits.
6. One-ERISC routers in scope; the relaxed flush (`FLUSH_RELAXED`) out of scope.

## Audit (agent; code reads against v4rxdas2, receiver unchanged since in its relevant parts)

- Every non-posted local data write (local write, scatter chunks, fused write, 1D sparse multicast hop) carries the
  slot's transaction ID; forwards carry it too (posted). Atomics and stateless forward doorbells are non-posted on
  write_at_cmd_buf (buffer 3) with tag 0, which lies in VC0's ID range. Posted inline writes get no ack: no flush can
  cover them, today included.
- A pending atomic kept as {address, value} in L1 and issued later as a whole `noc_semaphore_inc` cannot be disturbed
  by other buffer-3 users (unlike §63's staged command).
- Loop exits: E1 `receiver::run` ends on termination then `teardown()`; E1 parks in `end_round` during the base
  firmware's context switch / PAUSE / DRAIN / RETRAIN. One ERISC: `single::run` inner loop exits on router commands
  (PAUSE loop, re-entry) and the outer loop on termination.
- v4 on Blackhole always takes the global branch of `flush_write_to_noc_pipeline` without deadlock avoidance (VC0,
  VC1, VC2; one or two ERISCs) and the per-channel loop with it.

## Proposed design (agent; not reviewed)

State: one record per channel that delivers locally, in memory in the channel's table (VC0: `Tables.ch0`; VC1/VC2:
`ChannelState`; one ERISC: `ReceiverState`), so no loop-carried register is added:

```cpp
struct PendingAtomic {          // a flushed atomic waiting for its channel's earlier writes
    uint32_t unproven = 0;      // channel IDs not yet read with no write outstanding; 0: none waits
    uint32_t next = 0;          // next ID to read, offset from the channel's first (oldest first)
    uint64_t noc_addr = 0;  uint32_t value = 0;   // copied: the slot retires as usual
    bool waiting() const { return unproven != 0; }
};
```

Gate: a test at the start of each channel's step (2 instructions, 1 L1 load per running channel per pass when nothing
waits; cheaper than a per-pass mask on the usual single busy channel). Resolver out of line:

```cpp
// Earlier writes acknowledged: nothing of this ERISC outstanding now (one read, non-blocking), or each channel ID read
// at zero (an ID read at zero stays proven: the channel issues nothing while it waits).
bool earlier_writes_acknowledged(PendingAtomic& a, uint32_t first, uint32_t count);
__attribute__((noinline)) bool settle_atomic(PendingAtomic& a, uint32_t first, uint32_t count);   // proof, then issue
__attribute__((noinline)) void flushed_atomic_inc(PendingAtomic& a, uint64_t addr, uint32_t value, ...);  // fast path
// step<CH>, after the acks:
if (!t.atomic.waiting() || settle_atomic(t.atomic, C::Ids::first, C::Ids::count)) { progress = receive<CH>(...); }
release_completions(...);   // continues while waiting
```

Only the three flushed-atomic call sites change (standalone, fused, scatter last chunk); writes, scatter writes and
`deliver_other` stay untouched. Drain: a noinline `drain_atomics(tables)` after the loops in `receiver::run` and
`single::run` (not at E1's park: open question). `GroupIds` gains `first` and `count` (IDs the slots actually use).
Optional step B: move buffer 3's tag off ID 0 to an unused ID at init, so atomics and doorbells leave VC0's range.

Rejected: holding the slot (rings collapse); staging the atomic in buffer 3 (overwritable, §94); a global snapshot
(unsound); a global check only (starves); a per-pass mask gate (costlier on single-channel rows); inlined drain sites
(§63's code growth).

## Expectations (agent, hyp)

- Hot path: +2 instructions and +1 L1 load per running channel per pass; ~70 instructions of new out-of-line code.
- Rows that never flush: +0.5 to +2 cycles/packet (VC1 split / VC2 on: +1 to +3).
- Single-stream flushed rows: about flat (decision 3 keeps the channel waiting as long as today; §63's gain came from
  decoding the next packet during the wait, excluded here). Gains where something else shares E1: one ERISC with other
  traffic, VC1 split with flushes on one VC, rings and tori (first-level acks keep flowing): up to today's per-signal
  spin (220-290 cycles per signal on the v2 router, meas §64; v4's own cost unmeasured).

## Test plan (agent)

0. Offline: all configs build; size_report.py; spills on E1's loop.
1. Cheapest isolating A/B first: "gate only" (record + gate + drain, `deliver` unchanged) on rows that never flush
   (census suites 2D/1D/ring/torus, one ERISC, VC1 split). If it costs more than 1-2 cycles/packet, rethink first.
2. Full variant vs the lead: streams/flush_mix.yaml (fused every packet, chunks of 8/16 incl. 7 writes + 1 flushed
   fused, chunk-8 standalone atomic; E/W and N/S; flush on/off), flush_mix4.yaml, p3_mixes, one ERISC; new yamls:
   flushed scatter, bidirectional flushed stream on one ERISC, VC0 flushed chunks + VC1 writes. (The test kernel sends
   RELAXED; v4 treats it as strict.)
3. Step B (tag move) A/B.
4. Validation: variant_val_runs.sh (13), census_vc1_runs.sh, test_fabric_sanity_common.yaml (scatter atomic flush rows,
   not in the 13).
5. Ordering stress test: none exists (the test infra puts payload and semaphore on one core, which hides an early
   atomic). Sketch: signal i follows K stamped pages to a far core or DRAM; the consumer on a semaphore core near the
   router waits for sem >= i and reads the stamps back over the NoC; M distinct buffers; background NoC load; control
   with flush=false must find mismatches; variants on VC1, one ERISC, and a final atomic just before teardown.

## Open questions (for the user, at the feature pass)

1. Keep the global "nothing outstanding now" test as the non-blocking first check (a departure from decision 1;
   without it each standalone atomic reads 4-16 IDs and VC0 serializes on ID 0)?
2. Move buffer 3's tag to a spare ID (step B)?
3. RELAXED (treated as strict today) would take the deferred path: fine?
4. Drain before E1 parks for a context switch / PAUSE, or hold the atomic across the park?
5. Revisit decoding the next packet during the wait later (§63's gain)?

## Problems found along the way (code reads, not device-checked)

1. v4 today: an inline-write packet to L1 (`deliver_other` -> `noc_inline_dw_write`, spoof path) reprograms the local
   write command buffer (`NCRISC_WR_CMD_BUF` = 0, confirmed) to posted; later local writes would go out posted (a flush
   would hang or a proof pass early). No test yaml sends inline writes. Check independently.
2. Main's router: a VC1/VC2 flushed atomic waits on IDs 0..3 (VC0's), not its own (no global branch in main's flush).
3. With deadlock avoidance and three receiver channels, IDs run past 15 (8 per channel).

## Revision note (user, 2026-11-02): keep the current VC working too

The user: with deferral, not only do the other VCs stay unblocked, the current VC can do useful work as well, e.g. the
next packet's route word loaded ahead (TODO.md, N/S floor fix 2). The complication is storing the state correctly and
watching for transaction-ID completions (full acks, not just a flush); part of it is already in the completion logic.

Main session's analysis (to settle before any build):
- **WRONG (2026-11-02, user's clarification below): item 3 is not relaxed in this work.** ~~Item 3 above is relaxed.~~ While an atomic is pending, the channel may keep handling later packets.
  - The argument: a strict atomic promises only that writes before it landed first. A later packet's writes landing
    before an earlier deferred atomic don't break it.
  - A later atomic's own proof covers every earlier write, the earlier packet's included.
  - Deferred atomics sit in a small in-order queue per channel, in channel state (memory). Each is issued when its
    proof holds; proofs only grow, so they complete in order. When the queue is full, the channel stalls as today.
  - Write the argument down and check it against every fabric atomic type (unicast inc, fused write+inc, multicast
    variants) before building.
- **The proof is "acknowledged", not "sent".**
  - Completions use NIU_MST_WRITE_REQS_OUTGOING_ID(trid) (`writes_left` / `trid_writes_sent`: the data has left L1,
    so the slot can be reused).
  - A strict atomic needs NIU_MST_REQS_OUTSTANDING_ID(trid) == 0
    (`ncrisc_noc_nonposted_write_with_transaction_id_flushed`).
  - Reuse the completion logic's transaction-ID bookkeeping, but cover every ID the channel wrote since its last
    proven point, not only the current group.
  - Posted writes stay uncovered, as today.
- **Placement and budget.** Records and the prefetched route word live in channel state (memory). The headroom check
  flagged torus + VC2 with a flush-defer-sized record on S2-explicit, so design against the stable placement and gate
  on --headroom. Check pending atomics outside the pass's opening read group (already 4 reads), and only when one is
  pending.
- **Sequencing:** design together with the route-word prefetch (shared mechanism); after the receiver and sender
  passes.

**User's clarification (2026-11-02), which supersedes the relaxation above.** This work is the *strict* flush mode with
the stall removed:
- Item 3 stands: while an atomic is pending, the channel takes no new packet (no writes or forwards from later
  packets).
- E1 doesn't spin on the flush. Other VCs keep going, and the current VC does only useful non-ordering work: its acks
  and completions, and loading the next packet's header/route word ahead (reads only).
- One pending atomic per channel, so no queue is needed.
- Letting later writes go through is the *relaxed* flush mode (FLUSH_RELAXED). It is a separate, more complicated
  piece of logic and stays out of scope here.

What still holds from the analysis above: the proof is "acknowledged" (REQS_OUTSTANDING_ID), not "sent"; reuse the
completion bookkeeping, covering every ID the channel wrote since its last proven point; channel-state placement; the
--headroom gate; checking outside the opening read group; designing it together with the route-word prefetch.

## Design with the route preload (2026-11-02)

The strict flush-defer is designed together with the route preload in V4_ROUTE_PRELOAD.md (§1.4: the record, gate,
proof and drain; §4 predictions; §6 stage S3, after the preload lands; §7 open questions 4-7).
