# Slot allocator: staged, full-rate-depth design (rework 2026-11-04; D1-D4 decided; build started)

Replaces the first needs-first proposal (kept as ALLOCATOR_NEEDS_FIRST_v1.md). Evidence for the needs is in
SLOT_ALLOCATOR.md §12. Sketch arithmetic: slotalloc/staged_sketch.py.

## 1. Goal (user, 2026-11-04)

- **Decent defaults, not the measured optimum per case.** The best split depends on the traffic, which the host does
  not know: what VC2 carries, or whether VC0 or VC1 flows matter more on a split. A rule built from the router's shape
  alone cannot hit every measured best row, and per-config exceptions would make it unreadable.
- **Later tuning stays simple.** Tuning changes a need or the leftover order; the stages stay as they are.
- **Overrides come later.** The design only has to admit them (section 6). No override mechanism is designed now.
- **The code explains mechanisms, not measured numbers.** Each need's comment says what sets it. Perf numbers and
  known misses stay out of the allocator. A clear mechanism makes the trade-off visible, e.g. when VC1 needs more slots.
- **Tests:** a CPU-only golden table, and only if we find we need it.

## 2. Inputs and output

`allocate_slots(RouterShape)` is a pure function, as in step 1 (SLOT_ALLOCATOR §1). It uses only facts the host knows
without measuring anything:
- channel buffer bytes and slot size;
- the VCs built and their channels: worker, forwarding (one per direction), receiver;
- 1D or 2D;
- whether a receiver is on a ring or torus.

It returns a slot count per channel, each with the stage and reason that set it. Every count is logged.

## 3. Needs

**Definition.** A channel's need is the number of packets it must hold in flight to cover its credit loop at the
router's per-packet rate: about the loop's latency divided by the time per packet.
- **Slots, not bytes.** A slot holds one packet, whatever its size.
- **Worst case.** Small packets are router-bound and set the need. Large, wire-bound packets take longer each, so they
  need fewer slots, and the table over-provides for them.
- **Each need is that of the channel carrying the link alone.** In mixed traffic the channels on a link share its
  rate, so each needs less than its full need. That is why a proportional squeeze (stage 1) is a decent default.
- **When a channel needs more than it got.** It needs more when its traffic takes a larger share of the link than the
  mix assumes. Example: VC1 on a workload where inter-mesh flows dominate a link. Under a squeeze, VC1 then runs at
  about the fraction it was given. That case is what an override is for (section 6).

One named constant per VC x kind x 1D/2D. Each one's comment states its mechanism.

| channel | need | what sets it |
| --- | ---: | --- |
| VC0 worker, 2D / 1D | 20 / 24 | The worker's loop: a slot frees only after the router sends the packet and the far receiver's credit returns over Ethernet. |
| VC0 forwarding, 2D / 1D | 16 / 24 | The same loop, one hop on. **Provisional:** still improving at the top of the earlier sweep; re-measured on the port (§12.12, lb2 175938). |
| VC0 receiver, 2D / 1D | 16 / 32 | A slot frees only when its local write and its forward have left and its credit is sent back. |
| VC1 forwarding / receiver | 10 / 8 | As VC0's, for inter-mesh traffic: the knee with VC1 alone on the link. |
| VC2 worker / receiver | 4 / 4 | **Placeholder, unmeasured (user, 2026-11-04: start low).** VC2 is one-hop neighbor exchange with a delivery-only receiver; what it carries is unknown, so it starts near its minimum and grows by override. |

The forwarding need and whether needs hold across slot sizes (shown so far for the 2D worker only) are the two
measurements running on the port now (§12.12).

## 4. The stages

0. **Budget and minimums.**
   - N = floor(channel buffer bytes / slot size). This is the only place bytes appear.
   - Every channel's minimum is 2; a ring's VC0 receiver also keeps the bubble rule's 2.
   - The cap is 127 slots per channel.
   - If the minimums do not fit in N, the config is an error, as today.
1. **Receivers, in order first but not first in amount.**
   - The squeeze factor is f = min(1, N / sum of every channel's need).
   - Each receiver gets f x its need, rounded (decision D1).
   - Order, not amount: receivers taking their full need first would leave the senders the whole squeeze at large
     slot sizes. The 1/5 VC0 receiver floor did roughly that and cost the VC1 split's VC0 2-hop rows +24-32%
     (SLOT_ALLOCATOR §8, study tree). The cause, forwarding squeezed from 14 to 10, is a hypothesis.
   - Why receivers go first: these come from the receiver's count:
     - the power of 2 a ring needs (its index wraps with a mask; otherwise a compare);
     - the completion group size (<8 -> 1, <16 -> 2, else 4);
     - the transaction IDs that follow the groups;
     - the bubble threshold T = max(2, R - 16).
   - Fixing receivers first lets those rules read a final count. Whatever the rounding takes or frees, the senders
     absorb in stage 2.
**Router contract (added 2026-11-05 after a hang):** the router keeps one downstream forwarding ring size per VC
(router_downstream.hpp, `DownstreamTraits<VC>::num_slots`), so every forwarding channel of a VC must get the same
count. The first build split them by largest remainder per channel (16/15/15). It hung on its first data-checked runs
on two nodes (lb2 176405, 176425); the cause is a hypothesis until the fixed build runs clean. Fix (user, 2026-11-05):
- a VC's forwarding channels are one unit in stages 2 and 3;
- leftovers go to them only in whole rounds;
- rounding goes to the VC0 worker;
- a host TT_FATAL guards the contract.
The table in section 5 predates the fix; the gate output replaces it.

2. **Recalibrate, then the senders.**
   - B = N - sum of receivers; f' = min(1, B / sum of sender needs).
   - Workers and forwarding channels get f' x their need, by largest remainder.
   - A channel below its minimum is raised; the slot comes from the sender with the most above its own minimum.
3. **Leftovers** (as built, 2026-11-05): what is left after stage 2 (past every need, or the rounding a squeeze's
   units could not take) goes to the groups of the leftover order in turn, each in whole steps.
   - A group's receivers take one step: each to its next power of 2, all of them or none, and none past 32 slots
     (`Limits::max_receiver_slots_from_leftovers`, the deepest receiver measured). A ring keeps its power of 2.
   - A group's senders take +1 a unit at a time, unit after unit in the group's order and round after round, up to the
     cap (stage 2's units: each worker; a VC's forwarding channels together, so they keep one count). A group of one
     kind thus takes whole rounds.
   - A step that does not fit whole, or a group with no channels, passes what is left on to the next group.
   - The leftover order is data, with a default per dimension (user, 2026-11-05):
     - 2D: `[[VC0 receiver], [VC0 fwd], [VC0 worker]]`; the worker takes the rest.
     - 1D (line, ring, NeighborExchange): `[[VC0 receiver], [VC0 worker, VC0 fwd]]`, +1 on each in turn, worker first.
       Without forwarding channels (NeighborExchange) the worker takes the rest.
4. **Derived values.** The completion group size, the transaction IDs and the bubble threshold come from the final
   receiver counts by their own rules. The TRID rule is still open; it is re-measured on the port.

## 5. What it gives (as built: c53b5a5cbbd's forwarding units, tweaks (a) and (b), the 32-slot receiver cap, the 1D leftover order)

Rows read VC0 worker, forwarding, receiver; then VC1 forwarding, receiver; then VC2 worker, receiver. A VC's forwarding
channels share one count. f is the stage 1 squeeze factor (shown when below 1). From slotalloc/staged_sketch.py, which
matches the port's logged rows on every reconcile and coverage config.

| config | 4352 B (N 81 / 1D 83) | 3072 B (N 115 / 117) | 8704 B (N 41) | 15232 B (N 23) |
| --- | --- | --- | --- | --- |
| 2D mesh / torus | 17,16,16 (f 0.99) | 20,21,32 | 9,8,8 (f 0.50) | 7,4,4 (f 0.28) |
| 1D line / ring | 26,25,32 | 43,42,32 | 12,13,16 (f 0.51) | 7,8,8 (f 0.29) |
| NeighborExchange | 51,32 | 85,32 | 25,16 | 7,16 |
| VC1 split | 15,11,8 + 7,4 (f 0.68) | 19,15,16 + 9,8 (f 0.96) | 8,6,4 + 3,2 (f 0.34) | 5,2,4 + 2,2 (f 0.19) |
| VC2 on | 16,14,16 + 3,4 (f 0.90) | 18,19,32 + 4,4 | 8,7,8 + 2,2 (f 0.46) | 6,3,4 + 2,2 (f 0.26) |
| VC1 + VC2 | 13,11,8 + 6,4 + 3,2 (f 0.63) | 15,14,16 + 9,8 + 3,4 (f 0.90) | 7,5,4 + 3,2 + 2,2 (f 0.32) | 3,2,2 + 2,2 + 2,2 (f 0.18) |

Without the cap, (b) would raise the 1D / ring and NeighborExchange receivers at 3072 B to 64 (24,29,64 and 53,64).
Nothing measured a 64-slot receiver, and the bubble threshold was validated only to R 32. The user chose the cap
(2026-11-05), so those rows stay at c53b5a5cbbd's, which lb2 176714 measured. The 1D leftover order (user, 2026-11-05)
then moved the 1D / ring rows from 24,27,32 and 24,61,32 to 26,25,32 and 43,42,32; NeighborExchange, 8704 and 15232 B
do not change (no leftover past the receivers there, or no forwarding channel).

For orientation only (not a target), at 4352 B the port runs main's tier table and the study ran its own rows:

| config | port today | study |
| --- | --- | --- |
| 2D | 17,16,16 | 17,16,16 |
| 1D | 16,16,32 | 26,25,32 |
| VC1 split | 51,4,8 + 2,4 | 19,9,16 + 5,4 |
| VC2 on | 43,7,11 + 2,4 | 17,14,16 + 2,4 |

## 6. Overrides (later; what the design admits)

Each override replaces one input; the stages do not change.
- **Explicit rows:** skip stages 1-3. This is the study's switch semantics, and the port slot patch has it.
- **Needs or a ratio:** replace table entries, or scale one VC's needs (a ratio, like step 1's `vc_ratio`). The
  squeeze then follows.
- **Leftover order:** replace the list.

## 7. Decisions

- **D1. Receiver rounding: nearest power of 2 for every receiver** (user, 2026-11-04). One rule. It lands on the
  ring wrap and on the group thresholds (8, 16). Under a squeeze it rounds down hard: the VC1 split's VC0 receiver at
  f 0.66 is 10.6, which becomes 8 (group 2).
- **D2. Default leftover order: `[[VC0 fwd]]`** (user, 2026-11-04, after §12.12 on the port). The worker gains nothing
  past its knee (17-20); forwarding keeps gaining on N/S multi-hop (21-24) and turns (17-20). The VC0 forwarding
  full-rate depth stays 16. (Briefly `[[VC0 worker]]` before §12.12 was back.)
  Fallback (coordinator, 2026-11-04): the order is `[[VC0 fwd], [VC0 worker]]`. A group with no channels on a router
  passes the leftovers on, so a router without forwarding channels (NeighborExchange) gives them to the worker
  instead of leaving them unused (24 / 32 plus 27 unused at 4352 B, against the tier table's 32 / 32).
- **D3. VC2: 4 / 4 to begin with** (user, 2026-11-04), not 16 / 16.
- **D4. An allocator file names the constants** (user, 2026-11-04): tt_metal/fabric/builder/fabric_slot_allocator.
  {hpp,cpp}, one named constant per VC x kind x 1D/2D, named **full-rate depth** (user, 2026-11-04): the slots a
  channel needs to run at full rate when it alone carries its link. "Need" in this doc means the same thing.

**Device A/B and tweaks (2026-11-05):**
- **Results:** P150 lb2 176714 and Galaxy 176753 met the 1% bar on VC1 and VC2 (Galaxy medians -46% to -59%), 1D at 4352 B, ring and NeighborExchange. They missed on P150 2D 4352 B (forwarding rows +5.5% to +9%: forwarding 15 vs tier 16), 2D 3072 B (1-hop N/S +2.5% to +3.5%) and 1D 3072 B (+1.7%).
- **User decisions:**
  - (a) 2D VC0 worker full-rate depth 20 -> 18, inside its 17-20 knee, so 2D 4352 B equals the tier row 17,16,16.
  - (b) Leftover order: receivers to their next power of 2 first, only as a whole step, then VC0 forwarding in whole rounds, then the VC0 worker. This gives 2D 3072 B its best measured row, 20,21,32.
  - (c) The 1D 3072 B +1.7% is accepted.
  - The cap (user, 2026-11-05): leftovers raise a receiver no deeper than 32 slots, the deepest measured.

**2D 3072 B trade-off, accepted (user, 2026-11-05):** 20,21,32 against tier 35,16,32. 1-hop N/S is +2.1% / +3.0% and
3-hop E/W -0.5% / -3.1% (lb2 176854). An N/S router has no E/W forwarding channels, so its share of the budget sits in
the worker, and N/S rows want more worker slots than E/W ones. One chip-wide allocation can't serve both; a finer
per-router allocator (step 2) would. The 32-slot cap on leftover receiver raises is also in (user, 2026-11-05). 1D gets
its own leftover order, [[VC0 worker, VC0 fwd]] round-robin, giving 26,25,32 at 4352 B and 43,42,32 at 3072 B; a
P150 1D run and a Galaxy ring-jam sanity are pending.

## 8. Next

1. §12.12 on the port (lb2 175938: forwarding knee; whether receiver and forwarding needs hold across slot sizes).
   It settles the forwarding need and D2.
2. The user's decisions D1-D4.
3. Build on the port worktree with the slot patch's explicit-row switch kept as the override. Then A/B against
   main's tier table on the standard P150 sets, and on Galaxy for the VC1 and VC2 rows. Mixed rows are scored by link
   total. The acceptance bar is to be agreed; proposed: never more than 1% slower than main's tiers in any suite.
