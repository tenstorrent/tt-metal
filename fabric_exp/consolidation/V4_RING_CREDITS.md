# v4 credits by ring position: design note (L1 transport)

Status: not adopted (2026-10-29; RESULTS §98, v4rp-v4rs). The receiver half is worth -10 to -38 cycles/packet where
the receiver binds (v4rr); the sender's attribution under mixed traffic (a run per packet) costs more than that on
multicast rows. The lead is v4rs (v4tm's protocol). A register bit-history sender is open for the user's call.
Draft of 2026-10-28: user: go ahead with the L1 design first (the stream-register transport is a later, measured
alternative). Sources: RESULTS §98 ("Where the single-ERISC ring cycles go", job 131, and its correction; the v4f, v4u,
v4p and v4sc entries of 2026-10-27), `variants/v4tm` (the current lead).

## 1. Why

Under first-level acks (rings and tori) v4's receiver spends ~89 cycles/packet on credit bookkeeping (job 131, one ERISC;
the same code bounds two-ERISC rings on the receiver ERISC). v4 already made the transport cheap (one frame per pass,
~13, against main's busy check and register write per credit, ~51); what it pays for is attribution and counting (~50):
per arrived packet a header read of `src_ch_id`, ring arithmetic and an ack count store; per completed slot a source
lookup and a count store. The earlier lessons hold: credits cannot be held or batched under first-level acks (a held
completion is a bubble slot's credit: v4f2's half-ring and v4p's full-ring collapses), so the fix is cheaper events,
not fewer (v4u).

## 2. The idea

A VC's far receiver ring fills in the order its sender sends (one link, one TX queue per direction), and the receiver
acks in arrival order and completes oldest first. So acks and completions are prefix counts of the sender's sequence of
sends, and the receiver does not need to know which sender channel a packet came from:
- the receiver returns two absolute counts per VC, packets acked and packets completed (monotonic, unsigned, compared
  by difference as the counters are today), in one 16-byte frame;
- the sender records, per far ring slot, the channel that sent into it (one byte, written at the send, where the channel
  is a compile-time value), reads the two counts once per pass, frees far slots by the completion difference and credits
  producers by walking the newly acked slots of its record.

Nothing is held: each count goes out exactly when its credit would today, so the bubble rule ("on deadlock-avoiding
rings a completion must not wait for anything") stands.

## 3. Scope of the first build

Routers whose only VC is VC0 with first-level acks: rings and tori (deadlock avoidance; single-mesh here, so no VC1),
both ERISC modes, including mux mode on rings (one channel: attribution compiles away). Every other router is unchanged
(same images, checked offline). Later, measured separately: VCs without acks (meshes, 1D), where completions credit the
producers and would be attributed at the sender too (grouped); some of those rows are sender-bound.

## 4. Design

### 4.1 Receiver

- `acknowledge_arrivals`: consume the packets-received register by `arrived`, `available += arrived`, `acked +=
  arrived`. No header read, no loop.
- `retire`: no source record (`SlotRecords` and the header word's `src_ch_id` drop out).
- `release_completions`: when due and proved as today (one slot at a time under acks), `completed += n`.
- The frame: `[acked, completed]` in the first 16 bytes of the receiver's ack block, sent to the far sender's ack block;
  once per pass when either changed and the TX queue is free (`send_credits`, as today, 16 bytes instead of the whole
  counter block).

### 4.2 Sender

As built after the first build's measurement (v4rq; v4rp's sender, a per-slot record taken in a step of its own at the
pass's start, cost the sender +42 cycles/packet on light rows, job 133):
- Sends are recorded as runs, (channel, number of consecutive sends from it), oldest first, in the forwarding table: a
  send from the newest run's channel adds one to it, any other starts a run (`SendRuns::sent_from`). While one channel
  sends there is one run, so a send is +1 and an ack one credit, on the same code path as mixed traffic; the work grows
  only with changes of channel. At most one run per packet in flight (at most N = the far ring's slots, at most 255).
  A router with only the worker channel keeps just the two counts (`RingCounts`).
- `send<CH>`: no `src_ch_id` stamp (no producer stamps it either), no credit read after the send.
- The worker visit reads `[acked, completed]` in its existing load group with the connection semaphore (no separate
  early-exit test) and takes them for every channel (`take_ring_credits`): `free_slots += completed - taken`; the newly
  acked packets credited from the oldest runs, the worker's `acked += n` and the worker credit, a forwarding channel's
  `acked += n` and its upstream credit. A newer frame landing between the two loads (completions read ahead of acks) is
  taken as read; a later visit takes the rest.
- The forwarding visits send and record their sends; their `in_flight` bits are not needed for credits.
- Connection events keep using the worker's `acked` count (the credits returned so far).

### 4.3 Memory, init, teardown

The existing counter blocks (host allocation unchanged): the receiver writes words 0-1 of its ack block, the sender
reads words 0-1 of its remote ack block; zeroed at init as today (the host with two TX queues, the router itself with
one). Compile-time selection: `ring_position_credits = first-level acks on VC0 && no VC1 && counter credits`.

## 5. What moves

- Receiver, per packet: no header read, ring arithmetic or count stores for acks; no source record or lookup for
  completions; the frame is 16 bytes. Estimated ~45-50 of its ~89 (from job 131's lines; to be measured).
- Sender, per packet: a run update instead of the header stamp; per visit, two L1 loads in the worker visit's group (in
  place of the per-channel counter loads) and one step per run crossed. Ring configs, sender::run instructions / stack
  loads / stack stores: v4tm 401/36/46, v4rp 457/50/69, v4rq 405/41/50; receiver::run 517/19/33 -> 481/18/20.

## 6. Validation and measurement

1. Offline: every non-ring config loads the same image as v4tm; ring and torus configs compile (two-ERISC, one ERISC, mux
   ring).
2. Device, functional (data-checked): ring and Y-torus validation, val_mux, mesh / 1D sanity and the VC1 split
   (unchanged code), unit tests; both ERISC modes on rings.
3. Device, throughput, same job against v4tm, two passes: two-ERISC rings and Y-torus (exp_ring, exp_torus_y), one-ERISC
   rings (exp_ring_se), mux ring rows (exp_mux); full-ring and half-ring rows as the bubble check.
4. Galaxy later: the saturated full ring (the bubble behavior at scale) before it can lead.

## 7. Risks

- Order is load-bearing: the far ring's fill order and oldest-first completion become invariants of the protocol (stated
  at both ends in the code); a future out-of-order completion would break attribution.
- The sender reads its credits once per pass instead of in the visits; a pass with nothing outstanding must not read.
- Two ERISCs: the attribution walk lands on E0, which had slack on rings (v4u profile); meshes are out of scope here.
