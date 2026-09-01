# Cross-Node Dataflow Buffers: Unified Access-Pattern Model

This document proposes collapsing the CrossNodeDFB / PersistentDFB "sender flows"
(A/B/C/D) into the access-pattern vocabulary already used by the local
DataflowBuffer (`pap` / `cap`, `STRIDED` / `ALL`). It exists because the current
API asks a user to learn two unrelated models for what turns out to be one thing,
and because multi-hart senders on Quasar need a rule that does not fork from the
local-DFB rule.

Status: design proposal. Nothing here is implemented. The equivalences in §3–§5
describe today's *semantics* accurately; §8–§10 describe what would have to be
built.

Related:

- Local DFB host API: `tt_metal/impl/dataflow_buffer/dataflow_buffer.hpp`
- Cross-node host APIs: `cross_node_dfb.hpp`, `persistent_dfb.hpp` (this directory)
- Device APIs / current flow docs: `tt_metal/hw/inc/api/dataflow/{cross_node,persistent}_dfb.h`
- Prefetcher contract: `tt_metal/impl/buffers/prefetcher_matmul_design.md`

## 1. The problem

A local DFB is one FIFO. `num_producers` harts and `num_consumers` harts share it,
and the access pattern says how:

- `STRIDED` — participants partition the entries. Participant `i` owns entries
  `i, i+N, i+2N, ...`.
- `ALL` — every consumer sees every entry. Producer-side `ALL` is illegal
  (`add_dataflow_buffer` fatals on `pap == ALL`); it is a fan-out pattern, and
  fan-out only makes sense on the consuming side.

A cross-node DFB is documented completely differently, as four sender flows:

| Flow | Shape |
|---|---|
| A | `write_broadcast` — same bytes to every receiver, one collective credit |
| B | `write_to_receiver` per receiver — unique bytes, one collective credit |
| C | `reserve_back_for_receiver` / `push_back_to_receiver` — unique bytes, independent credits |
| D | `write_strided` — unique bytes from a contiguous staging buffer, one collective credit |

Nothing in that vocabulary tells you how *harts* share a sender, which is exactly
the question Quasar forces. The obvious patch — "sender hart `p` handles receivers
`r = p; r < M; r += P`, and collective operations stay single-hart" — introduces a
third concept that exists nowhere else in the stack.

## 2. The claim

The flows are not a separate axis. They are `ALL` and `STRIDED` observed at core
granularity, on a buffer whose consumers happen to sit on different nodes.

The *mechanism* necessarily differs, because L1 is not shared across nodes:

- Local `ALL` keeps one physical copy and lets the remapper fan credits to each
  consumer TC. Cross-node `ALL` must replicate the bytes on each node.
- Local `STRIDED` interleaves all columns in one physical ring. Cross-node
  `STRIDED` gives each node only its own column, contiguously.

The *semantics* are identical, and so the programming model can be identical.

## 3. Flow-to-pattern equivalence

| Today's flow | Equivalent configuration |
|---|---|
| A — broadcast | `cap = ALL`. One logical stream; every consumer reads every entry. |
| B — unique, lockstep credit | `cap = STRIDED`, collective `reserve_back` across all columns. |
| C — unique, per-receiver credit | `cap = STRIDED`, per-column `reserve_back`. |
| D — `write_strided` | `cap = STRIDED` plus a write helper for a contiguous source. |

Two observations fall out of this table.

**B and C are the same buffer, not different flows.** They differ only in the
granularity of the reserve call. Locally, a single producer hart that
round-robins over several consumer columns has exactly this choice: wait for
space on all of its TCs, or wait per TC. B is the collective form, C is the
granular form. Neither needs its own configuration.

**D is a write helper, not a topology.** `write_strided`'s staging layout
`[recv0][recv1]...[recvM-1]` is one round of `M` consecutive logical entries laid
out contiguously at the source. Naming it "strided" next to
`AccessPattern::STRIDED` is an unfortunate collision — the first describes the
*source* buffer layout, the second describes *entry ownership*. Under the unified
model they finally refer to the same thing, which removes the collision rather
than deepening it.

## 4. Layout and accounting

Let `E = num_entries` (logical stream depth), `P = num_producers` (sender harts),
`M` = number of receiver nodes.

### STRIDED consumers

Logical entry `e` belongs to node `e % M`, at that node's slot `e / M`.

```
logical stream:   e0  e1  e2  e3  e4  e5  e6  e7      (M = 4)
                   |   |   |   |   |   |   |   |
node 0 ring:      e0              e4                  contiguous, depth E/M
node 1 ring:          e1              e5
node 2 ring:              e2              e6
node 3 ring:                  e3              e7
```

Per-node L1 is `(E / M) * entry_size`. Summed over nodes that is `E * entry_size`
— the same total a local STRIDED DFB of depth `E` occupies. The distribution
changed; the accounting did not.

Note that each node's ring is *contiguous*, unlike local STRIDED where a column's
entries sit `stride_in_entries` apart. Cross-node STRIDED is the easier layout to
consume.

### ALL consumers

Every node holds a full copy of the stream.

```
node 0 ring:      e0  e1  e2  e3  e4  e5  e6  e7      depth E
node 1 ring:      e0  e1  e2  e3  e4  e5  e6  e7
node 2 ring:      e0  e1  e2  e3  e4  e5  e6  e7
node 3 ring:      e0  e1  e2  e3  e4  e5  e6  e7
```

Per-node L1 is `E * entry_size`; total is `M * E * entry_size`. `ALL` costs `M`x
the L1 of `STRIDED` for the same logical depth. That is the honest price of not
having shared L1, and users sizing L1 need it stated plainly.

### Capacity formula

The existing local formula generalizes without change. From
`compute_capacity_and_stride`:

- `STRIDED`: require `E % max(P, C) == 0`; `capacity = E / max(P, C)`.
- `ALL`: require `E % P == 0`; `capacity = E / P`.

With `C = M`, cross-node STRIDED gives a per-`(hart, node)` column depth of
`E / max(P, M)`. When `P > M`, each node hosts `P / M` producer columns of that
depth, so per-node storage is still `E / M` entries. The formula is already
correct for the cross-node case; only its interpretation needs documenting.

## 5. Worked example: the DRAM prefetcher

From `prefetcher_matmul_design.md` §3: the tensor's K dimension is split into
`num_blocks` blocks, and for each block every one of the sender's `M` receivers
gets one page of its own bytes.

Order the logical stream as `e = blk * M + r`. Then `e % M == r` and
`e / M == blk`: entry `e` lands on receiver `r` at slot `blk`. Textbook STRIDED
with stride `M`.

The prefetcher's per-block staging buffer holds `M` consecutive logical entries,
contiguous — which is precisely what `write_strided` consumes today, and
precisely what a local single-producer STRIDED push of `M` entries does. The
prefetcher has been using STRIDED all along; it just was not called that.

## 6. Where broadcast still earns a primitive

Collapsing Flow A into `cap = ALL` is a vocabulary change, not a deletion. The
multicast implementation underneath must stay, because it is a real hardware win:
one NoC transaction and one credit atomic can service all `M` receivers instead
of `M` unicasts.

Multicast is eligible only when all of the following hold:

1. Every receiver holds the ring at the **same L1 address**. PersistentDFB's
   lockstep mapping already guarantees this (it is the reason the mapping shares
   one allocation), and CrossNodeDFB's HEIGHT_SHARDED-over-`all_cores` allocation
   does too.
2. The receivers form a rectangle the NoC can address as a multicast grid.
3. Every receiver is at the **same ring offset**, which holds only while credits
   stay lockstep.

Point 3 is the one to enforce: mixing per-column crediting into an `ALL` buffer
silently disqualifies the multicast fast path, because receivers drift apart.
`ALL` should therefore reject the per-column reserve/push entry points.

Genuine `ALL` users: matmul `mcast_in0` (one activation block to a row of
workers), the small per-op tensors every core needs (bias, LN gamma/beta,
scalers), fan-out stages of all-gather, and data-parallel weight replication.

## 7. Multi-hart senders

Set `num_producers = P` with `pap = STRIDED`. Hart `p` owns logical entries
`p, p+P, p+2P, ...`, exactly as it would on a local DFB.

The receiver partitioning that the `r += P` proposal was trying to hand-write now
**emerges** from the composition. Entry `e` is produced by hart `e % P` and
consumed by node `e % M`, so when `M` is a multiple of `P`, hart `p` only ever
touches nodes `p, p+P, p+2P, ...`:

```
P = 2, M = 4

e:      0    1    2    3    4    5    6    7
hart:   0    1    0    1    0    1    0    1
node:   0    1    2    3    0    1    2    3

hart 0 -> nodes 0, 2      hart 1 -> nodes 1, 3
```

No new rule, no separate core-partitioning concept, and no special case for
collective operations. When `P > M`, the composition still works: each node's
ring is internally strided across the `P / M` harts that feed it.

For `ALL`, there is no core partitioning — every hart's entries must reach every
node — but parallelism is preserved: each hart owns a contiguous block of the ring
and multicasts its own entries. That is local `STRIDED producer x ALL consumer`,
unchanged.

### What the user has to know

This is the payoff for Quasar. With `pap = STRIDED` and `P` producers, each hart
pushes `E / P` entries, and `E % max(P, C) == 0` is validated at create. That is
the *same* rule users already follow for local DFBs — one rule instead of two,
and the "how many entries does my hart handle?" question has an answer that does
not depend on whether the buffer crosses nodes.

Consumers are unaffected by `P`. A receiver waits on its own column's depth and
never learns how many harts produced it.

## 8. Credits

This is the real implementation cost, and it is the reason the unified model is a
proposal rather than a refactor.

Today there is one `(pages_sent, pages_acked)` pair per receiver, and the sender
derives its write offset from `sent % ring_units`. That is a single producer
column. Two harts sharing it race on the derived cursor — which is what motivated
the "collectives stay single-hart" rule in the first place.

The unified model needs one credit stream per `(producer column, node)`:
`P * M` counter pairs instead of `1 * M`. On WH/BH these are L1 words plus NoC
atomics, so the cost is L1 footprint and reset time, not new hardware. On Quasar,
tile counters are same-node, so the remote credit remains a NoC atomic into the
receiver's counter; only the relay side gets real hardware counters.

Two follow-on items:

- `ALL` free-space is `min` over `M` acked counters, which is `O(M)` polling on
  the sender unless an aggregated counter is introduced. Worth measuring before
  committing to large `M`.
- The local `cap == ALL` validation caps `num_consumers` at 4. That is a remapper
  limit and must **not** be applied to the cross-node case, which has no remapper.
  The validation needs to be scoped by buffer kind.

## 9. Relay composition

The unified model does not change how relay works, and in particular it does not
make the cross-node `cap` become the relay's `pap`.

Relay is a credit bridge over the *same* L1. On the receiver node there are two
objects sharing one ring:

| | Cross-node object | Local relay DFB |
|---|---|---|
| L1 | the ring | same ring (`borrows_memory`) |
| Producer | sender harts (remote) | the DM |
| Consumer | the DM | TRISC |

So `relay.pap` is always `STRIDED` with the DM as the single producer — which is
required anyway, since `pap = ALL` is illegal. `ALL` lives only on `relay.cap`,
which is how you get "every unpacker sees every page" without an illegal producer
pattern. The cross-node `cap` describes how *nodes* divide the stream; the relay
`cap` describes how *TRISCs on one node* divide what arrived. They compose; they
do not copy.

One constraint the shared ring does impose: the two objects must agree on byte
layout. Cross-node STRIDED gives each node a contiguous column, which is
compatible with a relay whose producer writes contiguously. A relay `cap = ALL`
with packed per-producer TC blocks is compatible because the relay has exactly one
producer. Mixed layouts that would require reordering bytes are not expressible
with credit aliasing alone and would need a copy.

## 10. Where the abstraction leaks

Documented deliberately, because "cross-node DFB is just a local DFB" is true of
the programming model and false of the cost model.

1. **L1 replication.** `ALL` costs `M`x. Local `ALL` costs 1x. Same config, very
   different footprint.
2. **Credit fan-in.** `O(M)` counters to poll, versus one remapper.
3. **Uneven shards.** A local DFB has a single `entry_size`. Flow C's stated
   use case of uneven per-receiver shards does not fit; it needs padding to a
   uniform entry, or an explicit escape hatch outside the unified model.
4. **Ordering.** Remote consumers see data only after write-flush ordering
   (`flush_writes()` before the credit). Local TCs handle this implicitly.
5. **`num_entries` changes meaning.** Today `CrossNodeDFB`'s `num_entries` is the
   *per-receiver* ring depth (`ring_size() = entry_size * num_entries` is
   documented as bytes per core). Under the unified model it becomes the *logical*
   stream depth, with per-node depth derived as `E / M` (STRIDED) or `E` (`ALL`).
   This is a silent semantic change to an existing argument and is the single
   most dangerous part of the migration. Either rename the parameter or force a
   call-site audit.

## 11. PersistentDFB

PersistentDFB bundles three separable things. Keeping them separate clarifies how
much of it is prefetcher-specific.

**Durable credits and a checkpointed cursor** — general. Useful for any
producer/consumer split across programs: persistent kernels, pipelined ops,
producer relaunch while a consumer drains. Nothing about this is prefetcher-shaped.

**N disjoint 1:M groups sharing one allocation** — prefetcher-shaped. This exists
because HEIGHT_SHARDED L1 is lockstep: one `address()` is freed on every core in
the grid, and the matmul prefetch path needs one `fifo_start` across the grid.
`N` separate PersistentDFBs would mean either `N` different addresses (breaking
one-FIFO programming) or `N` rings on the full grid (`N`x L1). Importantly, this
is an **allocation and placement** concern, not an access pattern. Under the
unified model the `N` groups remain `N` independent buffers that happen to share
an allocation, each carrying its own `pap` / `cap`. It stays orthogonal.

**Mid-flight entry-size resize** — prefetcher-shaped, and the one place the
unified model needs real design work. With `cap = STRIDED`, changing `entry_size`
alters every column's geometry, and pad credits have to be published per column
rather than once. With a lockstep `ALL` group, resize stays uniform and the
existing snap-forward logic carries over. This interaction should be designed
deliberately rather than inherited.

## 12. Proposed API shape

Creation takes the same fields as a local DFB, plus the node topology:

```c++
CreateCrossNodeDFB(
    program, device,
    sender_core, receiver_cores,        // node topology: 1:M
    dfb::DataflowBufferConfig{
        .entry_size   = ...,
        .num_entries  = ...,            // LOGICAL depth; see 10.5
        .num_producers = P,             // sender harts
        .pap = STRIDED,                 // ALL remains illegal
        .num_consumers = ...,           // harts per receiver node
        .cap = STRIDED | ALL,           // partition nodes | replicate to nodes
    });
```

Device side, every producer hart calls the same three functions it would call on
a local DFB, and the runtime maps hart index to column:

```c++
dfb.reserve_back(n);
dfb.write(src, n);
dfb.push_back(n);
```

`reserve_back_for_column(c, n)` / `push_back_to_column(c, n)` stay as the granular
form (today's Flow C), legal only under `STRIDED`. `write_broadcast` and
`write_strided` remain as write helpers; the implementation filters destinations
by hart (identity when `P = 1`).

## 13. Practical takeaways (how to actually make this work)

Do not rewrite WH/BH CrossNode or Persistent. Treat today's device protocol as
`P = 1`. Teach one programming model; let architecture only change legal `P`/`C`
and the credit backing store.

1. **One kernel API on all archs.** Senders write
   `reserve_back(n); write_strided|write_broadcast|write_to_receiver; flush; push_back(n)`.
   Receivers write `wait_front(n); pop_front(n)`. No `r += P` in user kernels, no
   `#ifdef ARCH` in those loops. `n` is this hart's batch (on WH/BH that is the
   full fanout).

2. **Host is where `P` lives.** `Create` / `Attach` take `pap`/`cap`/`num_producers`/`num_consumers`
   like a local DFB. WH/BH: FATAL unless `num_producers == 1` and CrossNode
   receiver harts `== 1` (the DM). Quasar: `P > 1` legal. JIT passes `P` and hart
   id into the device class the same way local DFB passes TC slots.

3. **Hart-filter the write helpers, don't fork them.** `write_strided` keeps the
   `[R0][R1]…[RM)` staging layout. The loop becomes "for receiver `i`, if
   `i % P != my_hart` skip." `P = 1` is today's WH/BH loop. `write_broadcast` /
   `push_back` credit only owned columns; after all harts finish one iteration,
   every receiver has progressed by `n` — same as WH/BH.

4. **Don't redo credits on WH/BH.** Keep 1×M `(sent, acked)` and derived
   `wr = sent % ring`. That is the `P = 1` column. Quasar multi-hart needs disjoint
   columns: either `P` counter pairs per receiver, or `P` harts owning disjoint
   subsets of the existing `M` counters (`i % P == hart`). Never share one `sent`
   across harts.

5. **Keep today's `num_entries` as per-receiver depth** on the existing Create
   signatures so prefetch and tests don't silently change meaning. If you need a
   local-DFB-shaped logical depth later, add it as a derived field
   (`logical_E = num_entries * M` for STRIDED, `= num_entries` for ALL) rather
   than reinterpreting the argument.

6. **`cap` chooses payload, not a second flow enum.** `ALL` → same bytes, lockstep
   credit, multicast when the grid allows. `STRIDED` → unique bytes per node;
   `write_strided` / `write_to_receiver` are how you fill columns. Per-column
   reserve/push stays for head-of-line avoidance. Do not put `ALL` on
   `pap`. Do not apply the local remapper `num_consumers <= 4` cap to CrossNode.

7. **Relay stays a local DFB.** CrossNode `cap` = how nodes split the stream.
   Relay `pap = STRIDED` (the DM), relay `cap` = how TRISCs on that node split
   what arrived. Never assign `relay.pap = cross_node.cap`. WH/BH relay remains
   1 DM; Quasar ALL-unpackers are relay `cap = ALL`.

8. **Persistent: don't touch mapping/resize/checkpoint for this.** N disjoint
   1:M groups are allocation. Durability is credits that survive programs.
   Prefetch stays one writer RISC + `write_strided`. On Quasar, launch that same
   kernel on `P` DMs; the DFB implementation partitions `M`. First version of
   resize stays lockstep / single-column as today.

9. **Implement in this order.** (a) Document + host validate `P==1` on WH/BH
   without changing device. (b) Make write/reserve/push take `my_hart`/`P` with
   the `P=1` path bit-identical to current WH/BH. (c) Quasar: per-column credits
   + `P>1` tests with the *same* kernel source as (b). (d) Only then attach
   `pap`/`cap` to Create as the public way to set `P`/`M` behavior.

## 14. Open questions

- Does `num_entries` become logical (consistent with local DFB, but a silent
  semantic change) or stay per-node (no migration risk, but the capacity formula
  no longer reads the same as the local one)? §10.5.
- Is `O(M)` acked-counter polling acceptable for production `M`, or is an
  aggregated credit counter required before `ALL` scales? §8.
- Do uneven per-receiver shards need first-class support, or is padding to a
  uniform `entry_size` acceptable for every real user? §10.3.
- Should mid-flight resize be supported under `cap = STRIDED` at all, or
  restricted to lockstep `ALL` groups in the first version? §11.
- Quasar: confirm there is no remote tile-counter post path, which would keep
  cross-node credits software-only and make the local relay the only TC consumer.
