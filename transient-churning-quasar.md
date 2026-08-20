# Quasar: move worker-completion signalling onto the dedicated fast dispatch signals

## Context

On Wormhole and Blackhole a worker core announces that it has finished a program by
incrementing a NOC overlay stream register on the dispatch core, and the dispatch kernel spins
until that register reaches an expected total. Quasar has no overlay stream registers, so the
Quasar bring-up substituted a region of the dispatch core's L1 —
`WORKER_COMPLETION_SEMAPHORES` (`tt_metal/impl/dispatch/command_queue_common.hpp:36`). Workers
reach it with a NOC atomic increment; the dispatch kernels poll it.

Quasar has purpose-built hardware for this handshake instead: a sideband of dedicated go and
done wires between dispatch-engine tiles and Tensix tiles, called the "fast dispatch signals"
(FDS) block in-tree. Moving the done direction onto it removes one NOC atomic per worker per
program launch from the critical path.

Scope, as directed: **the done direction only**. The go signal keeps travelling as a NOC
multicast into worker mailboxes. Behaviour must be unchanged — sub-device accounting, two
command queues, trace capture and replay, events, active-Ethernet accounting, and every "wait
for workers" path. The interim `TT_METAL_TENSIX_DISPATCH_CORES=1` path keeps the existing NOC
atomic mechanism.

This plan is **conditional**. No dispatch implementation starts until Gate 0 settles the FDS
register semantics and Gate 0b identifies a completion fence that orders every issuing worker hart's
writes before sideband done. A passing simulator test is necessary but is not evidence that silicon
has the same semantics.

**Gate 0's transport question now passes.** On a new simulator build, as of 2026-08-20, a go crosses
from the dispatch-engine tile to every Tensix tile, a done crosses back, and the dispatch side
aggregates dones across tiles. An earlier revision of this plan was parked on the opposite finding:
on the previous build nothing crossed in either direction. Nothing about the software changed, and
none of the register-interface findings changed either.

**Gate 0's remaining items have now also been run.** The same-group re-arm cycle works: a sink-side
inbox clear holds against a live source, the go de-asserts, and a second done for the same group
produces exactly one new credit. So the P1 protocol this plan is built on is viable, and the explicit
device-visible epoch that would otherwise have been needed in every go is not. The lane mapping is
measured for two tiles — lanes 0 and 4 — which is enough to show one lane per tile but not enough to
establish the assignment rule.

What remains is Gate 0b, the completion fence, which has not started and cannot be settled on a
simulator; the lane-assignment rule for topologies beyond two tiles; and the two-CQ ownership model.
See "Gate 0 result" below.

## Decision summary

| Question | Answer |
|---|---|
| What does FDS replace? | Tensix worker→dispatch transport on eligible launches. The cumulative 32-bit counting model remains the public interface. |
| Which hardware register drives it? | `GROUPID_STATUS` — a live, read-only, per-lane mask, not gated by the enable register, and now observed asserting under a real go. The engine-side field named dispatch instance 0 as the source. Its behaviour on deassertion and on re-arm within a group is still untested. |
| Which processors can talk to FDS? | **Data-movement cores only.** Gate 0 established that the Tensix engine processors cannot reach the interface at all, so the tile-wide owner must be a data-movement core and worker completion cannot be signalled from a compute kernel. |
| Who talks to FDS? | Exactly one **tile-wide** owner for all CQs on a dispatch-engine tile. A per-CQ `dispatch_s` owner is invalid unless hardware proves queue-selective isolation and routing. |
| Who writes the cumulative L1 words? | The tile-wide owner only. Clears, virtual-worker credits, and any non-FDS contributions must be serialized through it. |
| Can `WORKER_COMPLETION_SEMAPHORES` be deleted? | **No.** It remains the persistent interface consumed by dispatch waits and host-generated commands. |
| Gate | A worker-visible transport mode selected only when the resolved dispatch core is `CoreType::DISPATCH` and every completion producer is supported. `ARCH_QUASAR` is insufficient. |
| P2 fallback? | **Not needed on this build.** P1 was made repeatable: see "Gate 0 result". Should it prove unrepeatable on other hardware, the rule stands — stop and redesign an explicit shared epoch protocol rather than silently reducing the sub-device limit. |

### Review disposition

The direction — convert ephemeral FDS state into persistent L1 counters — survives review. The
original per-CQ owner, plain RMW publication, call-order swap, P2 fallback, and `dm.cc`-only
scope do not. They are replaced below with a tile-wide owner, owner-mediated clears, staged
configuration, a worker-visible transport gate, and a fully tested same-ID epoch protocol.

---

## Gate 0 result: the register interface is proven and so is the transport

Gate 0 has been run on the `emu-quasar-2x3_DISPATCH` simulator config — core descriptor
`quasar_simulation_2x3_arch.yaml`, two worker tiles, native dispatch-engine cores — under slow
dispatch, using
`tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp`, twice: against the previous simulator
build with the instrumented kernels described in change 1, and against a new build with the reduced
kernels that are in the tree now.

The register interface is real, complete, and characterised in detail; that work was done on the
previous build and stands unchanged. **On the new build both directions of the sideband carry.** The
register-interface findings below were measured before the transport worked and were not contradicted
by it, which is worth knowing: they were the reason the silence could be attributed to configuration
rather than to software.

### The register interface

**It is reachable only from data-movement cores.** This answers open question 7. Data-movement
cores are built as coprocessor-equipped parts (`-mcpu=tt-qsr64-rocc`) and the Tensix engines are
not (`-mcpu=tt-qsr32-tensix`), so the custom instructions the accessors use reach a different unit
on an engine processor. On all four engine-processor roles across both tiles, two registers with
different declared widths return an identical value, and that value advances by exactly twenty
from one processor's report to the next — the untouched destination register of an instruction
that did nothing, still holding a print-buffer offset. The same probe on every data-movement core
returns the declared field widths.

This constrains the design directly: the tile-wide owner must be a data-movement core, and worker
completion cannot be signalled from a compute kernel.

**Nine address bits are decoded**, a 512-byte window, and the dispatch map's documented `0x200`
base is stripped by that decode. Everything above `0x1FF` aliases back down. Thirty-four probe
points across both core types fit this model exactly, including aliases at `0x400`, `0x800`,
`0x1000` and `0x2000` landing on registers whose field widths match.

**There is one block per tile, shared by every data-movement core on it**, and it matches the tile
type: the dispatch map on a dispatch-engine tile, the engine map on a Tensix tile. This answers
open question 3, though not the way an earlier revision of this section claimed: configuration and
status are per **tile**, not per processor.

Measured directly. Every processor stamped its own index into an otherwise unused register early
and read that register back at the end, once all had written. On the dispatch tile all eight
processors read processor 4's stamp; on each worker tile all six read processor 2's stamp. The
processor that wrote last read its own value and all the others read that same foreign one.

This turns the duplicate-reader argument for a tile-wide owner from a policy choice into a
hardware constraint. Two processors on a tile touching FDS will overwrite each other's
configuration and consume each other's status, because there is only one set of registers.

**There is no per-processor banking.** The `0x1000` offset aliases rather than selecting a bank,
so `CORE_OFFSET` in the shim is not a stride for this interface. There is no other processor's
block to reach: every processor on a tile addresses the same one.

**Field widths match the register description exactly** — four bits for the inbox and outbox
registers, three for the engine-side group enable and status, eight for the count thresholds and
counts, thirty-two for the filter and the dispatch-side enable and status.

**Status and count registers are read-only**; writes to them are ignored. Status is **not** masked
by the enable register.

**Group zero is the idle value.** A lane presenting nothing reads as group zero, so the group-zero
status register is a live map of quiet lanes. The dispatch engine reports all thirty-two done
lanes quiet, and every engine block reports all three go lanes quiet. This partially answers open
question 1 — status is a live per-lane mask, not a latch, with no read side effects observed — and
corroborates treating group id zero as reserved.

### The sideband

Five results, all from the 2026-08-20 run. The full status lines are in
`quasar-fds-sideband-findings.md`.

**Both directions carry.** One worker latched a go and answered with a done that the dispatch engine
counted. This is the result the whole plan was waiting on.

**Dones aggregate across tiles.** Two worker tiles signalling the same group produced a done count of
two. This is the load-bearing measurement, because it is the one result the
uninitialised-destination-register problem cannot fabricate: a dead read instruction can return a
plausible 1 — the group id and the last value written both happen to be 1 — but not a 2 arising from
two separate tiles. Treat any future result that rests on a count of 1 as weaker than it looks.

**`GROUPID_STATUS` asserts, and names its source.** The engine-side three-bit field read 1, so
dispatch instance 0 drove the go. This is the register the owner publish loop is built on, observed
under a real assertion for the first time.

**The go wire is shared across groups, and the group filter is what separates them.** With half the
tiles in group 1 and half in group 2 and only group 1 signalled, a group 2 tile saw group 1's value
in a raw inbox register and did not latch it: its own group status stayed zero. This is the
mechanism the group-per-sub-device mapping depends on, and it is now measured rather than assumed.
It also means a foreign group's value in a raw inbox is normal and is not a leak — only the group
status says whether a go was accepted.

**An unsignalled group accumulates nothing.** Group 2 was configured with the same lane mask and a
threshold of one, so a single stray done would have satisfied it. Its count stayed at zero.

**Consecutive epochs of one group are distinguishable, so P1 stands.** `round1 count=1, count after
inbox clear=0, after settle=0, round2 count=1`, and on the worker side `round1 go=1, go de-assert
seen=1, group status after de-assert=0, round2 go=1`. Every step of the cycle set out under
"Distinguishing consecutive epochs" below holds: the sink-side clear survives a live source, the go
de-asserts and the group latch follows it back to idle, and the second done yields exactly one new
credit rather than two or none.

**Each tile drives one lane, and two tiles landed four apart.** `lane 0 carries group 1 -> core 0-0`,
`lane 4 carries group 2 -> core 1-0`, with group 0's status reading `0xffffffee` — exactly bits 0 and
4 clear. Two readings from opposite sides of the aggregation logic naming the same lanes. Per-group
counts agreed with the lane scan and every undriven group counted zero, which is the done-direction
isolation result the handshake tests could not produce.

The stride of four fits a lane space of four per tile — four engines per cluster, eight tiles,
thirty-two lanes — with software driving only the first of each four because the outbox is per tile.
If that holds it caps a dispatch tile at **eight** worker tiles rather than thirty-two, which the
mask model needs to account for. Two data points do not establish it; a descriptor with more tiles
would.

The handshake runs with the deglitch filter at zero, its reset value.

### What the previous build's silence eliminated

Kept because it says which explanations are already excluded if the symptom ever returns. On the
previous build, with the dispatch engine holding group 1 in its outbox for entire runs and twelve
data-movement blocks across both worker tiles driving done, every raw inbox on both sides stayed at
zero and neither quiet-lane map ever changed.

| Candidate cause | How it was eliminated |
|---|---|
| Timing races, lost edges | Both signals are held rather than pulsed, with tens of seconds of simulated overlap |
| Count thresholds | Set to one, and the raw inboxes bypass counting entirely |
| Enable masks, group id choice | The raw inboxes sit before aggregation |
| Deglitch filter | Swept 0, 1, 2, 8 and 64 on both sides against held signals |
| Stale auto-dispatch or interrupt state | Both read as zero before being explicitly zeroed |
| Wrong processor | All twelve user data-movement cores across both worker tiles |
| Instrument error | Field-width truncation and cross-address probes prove real registers and correct addressing |

The diagnosis at the time was that the lanes were not connected in that configuration, and that it
was a hardware-configuration question rather than a software one. A new build confirmed it. None of
the causes above is a live suspect, so if the sideband goes quiet again, suspect the build first.

### What to put to the hardware configuration owners

Two of the four original questions are answered by the new build: the lanes are instantiated and
connected in this configuration, and no block-level enable outside the two register maps is needed
to make them work. Still outstanding:

1. What does each of the 32 done lanes correspond to — one per tile, or one per Tensix engine? The
   block is named for the engine, but there is one block per tile with a single outbox register in
   it, which points at one lane per tile. This determines the worker-to-bit mapping.
2. The authoritative register specification: access types, whether status latches, deglitch threshold
   units, level versus pulse on the outbox, and clear/re-arm semantics. The generated headers give
   addresses, widths and reset values and nothing behavioural.
3. Whether silicon shares this build's ordering, clear and ownership semantics.

### Confidence in these findings

High on the register interface and the reachability finding. Each rests on a measurement shaped so
that the alternatives produce visibly different answers, and each was reproduced across processors
and tiles.

High on transport in both directions, on the strength of the two-tile done count.

Lower on everything the transport now makes testable but which has not been tested: lane mapping,
re-arm, deassertion, done-direction isolation, and anything beyond two groups or two tiles. A single
successful epoch is not a protocol.

---

## The design

### Why the owner must be tile-wide

Both `cq_dispatch.cpp` (dispatch_d) and `cq_dispatch_subordinate.cpp` (dispatch_s) consume the
persistent completion count. FDS observation is ephemeral and, under P1, destructive. Therefore
only one processor may convert a physical assertion into a software credit.

That uniqueness requirement crosses the command-queue boundary. Quasar co-locates both CQs on
one dispatch-engine tile and creates one `dispatch_s` kernel per CQ. Both kernels remain alive
while idle and spin in `cb_acquire_pages_dispatch_s`. If both can see the same FDS window, adding
polling there lets one worker assertion increment both CQ-local L1 slots. Host-only
`CQOwnerState` prevents concurrent enqueue but is invisible to the device kernels and does not
stop the inactive CQ from polling.

The production design therefore needs one tile-lifetime FDS service:

1. Exactly one processor owns dispatch-side FDS configuration, reads status/inboxes, and clears
   them.
2. It holds a device-visible `(active_cq, generation)` for each sub-device.
3. It publishes each physical done only into
   `completion_counter_offset(active_cq) + sub_device_index`.
4. Ownership transfer is an ordered device command acknowledged by the owner; host ownership is
   not considered transferred until that acknowledgement.
5. The service outlives each per-CQ command stream and exits only after every CQ has shut down.

A per-CQ implementation was to be reconsidered if Gate 0 proved that processor windows are
independently configured. **That door is now closed.** Gate 0 established that the whole tile
shares one register block, so processor windows cannot be configured independently and two
processors cannot avoid clearing each other's state. The tile-wide owner is a hardware requirement,
not a design preference.

`update_worker_completion_count_on_dispatch_d` (`cq_dispatch_subordinate.cpp:283-306`) remains
useful precedent for where liveness hooks belong. It is not precedent for ownership: its body is
a no-op on the co-located Quasar path.

### The owner state machine and publish loop

Per sub-device, the tile owner keeps:

- active and pending worker masks plus a configuration generation;
- active CQ and ownership generation;
- `fds_credited_mask`;
- epoch state: `QUIESCENT`, `COLLECTING`, `DONE_HELD`, or `CLEARING`;
- per-CQ cumulative L1 publication state.

While `COLLECTING`, it does:

```
raw     = FDS_INTF_READ(GROUPID_STATUS[group(sub_device)]);
status  = raw & active_worker_mask[sub_device];
newbits = status & ~fds_credited_mask[sub_device];
if (newbits) {
    fds_credited_mask[sub_device] |= newbits;
    owner_publish_delta(active_cq, sub_device, popcount(newbits));
    if (fds_credited_mask[sub_device] == active_worker_mask[sub_device]) {
        state = DONE_HELD;
    }
}
```

The unmasked `raw` value is retained for diagnostics; masking first makes an outside-mask assert
tautological. Duplicate held levels are expected and are suppressed by `fds_credited_mask`, not
treated as an assertion failure.

The owner is the only writer of the public L1 words. `CLEAR_MEMORY` on a completion word becomes
a clear-generation request to the owner; dispatch_d waits for the matching acknowledgement
instead of storing zero itself. Virtual-worker credits and any supported non-FDS producers are
also submitted to the owner. A plain CPU `+=` is permitted only after this single-writer
invariant is enforced. Otherwise `load N → independent clear/atomic → store N+k` loses an update.

Partial progress remains visible at the owner's polling cadence, not instantaneously as with a
worker NOC atomic. The owner must poll at the top of its service loop and in every blocking loop,
including page drain and shutdown. Telemetry must read the owner-published L1 words; its current
stream-register implementation is not behaviourally equivalent on Quasar.

### Distinguishing consecutive epochs — P1 or stop

A worker's done is a held 4-bit value. The in-tree helper explicitly says
`fds_clear_done()` is required between done signals of the same group. Dispatch-side inbox clear
and worker-side output clear are different operations; the original plan incorrectly relied on
the first while configuring only an init-time version of the second.

P1 is accepted only if Gate 0 proves this complete cycle:

1. Worker output is 0 before the first epoch starts.
2. At completion, the worker executes the proven completion fence, compiler barrier, then writes
   group `g`.
3. The owner credits every worker and enters `DONE_HELD`.
4. Before another GO, the owner clears each dispatch receive inbox and verifies that status stays
   zero for longer than the deglitch interval while the worker still drives `g`.
5. Only after zero is stable does the owner reset `fds_credited_mask`, enter `QUIESCENT`, and
   send the next GO/reset/replay signal.
6. On accepting every acknowledgement-producing GO/reset/replay signal, the worker writes
   `fds_clear_done()` before executing the epoch.
7. At completion it writes `g` again, and the owner must produce exactly one new credit.

If sink clear immediately re-latches while the source still drives `g`, or a same-value second
epoch cannot be made repeatable, this design stops. Ping-pong IDs are not a ready fallback:
phase is not carried in the go message, is not shared across CQs, and is not transferred during
ownership changes. A future P2 design would require an explicit device-visible epoch in every go
transaction and separate approval for reducing eight sub-devices to seven.

### Group-id budget and the command-queue dimension

The formulas are deliberately separate:

```
group_id = sub_device_index + 1                       // 1..8
l1_slot  = completion_counter_offset(active_cq) + sub_device_index
```

The worker must not use `dispatch_message_offset + 1`: that offset contains the CQ term, and
CQ1/sub-device7 would produce 16, which truncates to reserved id 0 in the 4-bit FDS field.

Sharing one group per sub-device is correct only because the tile-wide owner has an acknowledged
device-side ownership generation. Host `CQOwnerState` alone is insufficient. If the owner cannot
demultiplex a shared group to an active CQ without ambiguity, the combination of two CQs and
eight sub-devices does not fit 15 usable IDs and the change must not proceed.

The hardware premise underneath that mapping — that groups are isolated from each other — now has a
test, `DispatchEngineSubDeviceGroupIsolation`. Two disjoint sets of worker tiles take group 1 and
group 2, the dispatch engine sends a go for group 1 only, and the test asserts that group 2's tiles
never latch a go for their own group and that group 2's done count on the dispatch side stays at
zero while group 1's reaches its full total. A group 2 tile that sees group 1's value in a raw inbox
without latching it is reported rather than failed, since the go wire may be shared across groups
and in that case the group filter is the only thing separating them — which is exactly the mechanism
the sub-device mapping would be relying on.

Three things it does not establish. It exercises the FDS group mechanism, not the sub-device API:
nothing connects a `SubDevice` to a group id until the dispatch work in changes 4 through 7 exists,
so the worker sets are stand-ins. It does not test the done direction under load, because a quiet
group's workers never see a go and so never drive done — proving that a group 2 done cannot credit
group 1 needs a worker that drives done unprompted, which the reduced test deliberately no longer
does. And it covers two groups, not the full budget of eight.

---

## Changes, with confidence

Confidence is about *my analysis being right*, not about the change being easy.

### 1. Bring-up tests first — before any dispatch code. **Done, and blocking.**

This exists as `tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp` under slow dispatch, with
two kernels: a dispatch-engine sender and a data-movement worker. The dispatch engine sends a go and
waits for done; each worker waits for the go and answers with done.

One data-movement core per *tile*, because Gate 0 established that a tile has a single register
block shared by all of its data-movement cores, so two cores on one tile would overwrite each
other's configuration and consume each other's status. Separate tiles have separate blocks, which
is why fanning out across the grid is meaningful where fanning out within a tile is not. Three
tests use that, sharing one launch path and differing only in how the worker tiles are grouped:

- `DispatchEngineSingleWorker` — the minimal one-to-one handshake.
- `DispatchEngineAllWorkers` — fan-out to every worker tile, which is the one that can fail on a
  wrong mask or lane mapping. See Gate 1.
- `DispatchEngineSubDeviceGroupIsolation` — half the tiles in group 1 and half in group 2, with a
  go sent for group 1 only. See "Group-id budget" for what it does and does not establish.

Two further tests are the Gate 0 experiments the working transport made possible, both now run and
passing. They use their own kernel pairs, because each runs a different protocol rather than a
differently grouped handshake:

- `DispatchEngineSameGroupReArm` — two epochs of one group, with the receive inboxes cleared while
  the worker is still driving the first done, and the go de-asserted and re-asserted between epochs.
  The two counts either side of that clear are the P1 measurement. Kernels:
  `quasar_dispatch_engine_rearm.cpp`, `quasar_fds_worker_rearm.cpp`.
- `DispatchEngineLaneMap` — no go at all. Every tile drives a done carrying its own group id, and
  the dispatch side's raw inbox registers name the lane each value arrived on. Kernels:
  `quasar_dispatch_engine_lane_map.cpp`, `quasar_fds_worker_drive_done.cpp`. It also carries the
  done-direction isolation check, since every group is enabled on every lane and a group no tile
  drove must still count zero.
- `DispatchEngineWriteOrdering` — the Gate 0b harness. The worker writes a payload to the dispatch
  core's L1 over the NOC and drives its done; the dispatch engine reads the payload on seeing the
  done. Two arms, with and without a barrier before the done. Kernels:
  `quasar_dispatch_engine_ordered_read.cpp`, `quasar_fds_worker_ordered_write.cpp`.

The other tests do not exercise write ordering at all, and it is worth being explicit about why: in
them the worker writes status to its own L1, which is a local store with nothing in flight, and the
host reads it only after the program completes rather than on observing a done. They establish that
the signal arrives, not that data the signal announces is visible.

The test that produced the Gate 0 findings was larger, and deliberately so. It ran on every
data-movement core of both worker tiles and the dispatch tile plus every Tensix engine processor,
printed configuration readbacks, dumped every raw inbox, swept the deglitch filter across five
settings mid-wait, and drove done part way through the worker's wait whether or not a go had
arrived so that each direction reported independently. It also carried a probe header that
distinguished a real register from plain storage and from a custom instruction that does nothing,
and a per-processor stamp that settled the one-block-per-tile question.

That apparatus has been removed now that its questions are answered. The three tests in the tree
prove transport and report a readable result, but they do not reproduce the register-interface
evidence. Restoring instrumentation is a prerequisite for the next experiments, not just for
re-opening old ones; the techniques and the commit that holds them are recorded in
`quasar-fds-sideband-findings.md`.

Of the original matrix, these are settled:

- explicit disable and readback of interrupts and auto-dispatch — both read zero before being
  zeroed;
- read visibility from every candidate owner processor — all of them share one block per tile, so
  every processor sees the same registers and there is no isolation to establish;
- which processors can reach the interface at all;
- transport in both directions, aggregation of dones across tiles, and `GROUPID_STATUS` asserting
  under a real go;
- go-direction group isolation, on a shared wire separated by the group filter.

These are now possible and not yet done. They were previously blocked on an assertion arriving:

- the same-ID two-epoch P1 cycle — **done and passing**, `DispatchEngineSameGroupReArm`. This is the
  result that lets changes 5 and 7 be written as designed;
- done-direction isolation — **done and passing**, folded into `DispatchEngineLaneMap`;
- deassertion, and `GROUPID_COUNT` behaviour beyond a simple increment — **done** for the go
  direction and for count-versus-lane agreement;
- the physical lane mapping — **measured for two tiles, rule not established**. Lanes 0 and 4, one
  lane per tile, cross-checked against the group-0 status map. Two tiles is all
  `emu-quasar-2x3_DISPATCH` has, so this waits on a simulator build for a larger config;
  `DispatchEngineLaneMap` maps up to fifteen tiles in one run with no changes. Harvested layouts are
  still untried;
- routing to every dispatch instance, which needs a descriptor with more than one dispatch tile.

The shim gained the two accessors these need: a dispatch-side `fds_read_group_status`, and
`fds_read_neo_status` for the raw inbox, restored now that a consumer exists.

Gate 0 was to choose status-mask versus inbox polling and decide whether P1 exists. It has now
answered the first — status asserts, is live, and names its source, so the status-mask model in the
publish loop stands — and left the second open, because P1 needs two epochs and every test runs one.

*Side effect:* none — additive.

### 2. Make the vendored shim usable. **Done.**

- `overlay/fds_functions.hpp:10-11` — the two bare includes now name `"meta/fds_registers/..."`.
  This resolves by the quoted-include rule, so **no change to `qa_hal.cpp` `includes()` was
  needed**, which was the cheaper of the two fixes.
- `fds_clear_neo_status` used `_REG_OFFSET` (0x004) where every other dispatch-side helper uses
  `_REG_ADDR` (0x204). **This was not a defect, contrary to the earlier reading here.** Gate 0
  established that only nine address bits are decoded, so bit nine is discarded and both forms
  reach the same register. It is now consistent for readability, but nothing depended on it and it
  was never writing into the worker-side map.
- Both `fds_config_auto_dispatch` implementations skipped the enable-register write when disabling,
  so passing `false` left whatever the register already held. They now write it unconditionally.
  Stale state is one of the conditions this initialization is meant to survive, and Gate 0 read the
  register as zero before zeroing it explicitly.
- The helpers are one-line register accesses, so they are now `inline` in the header, and
  `fds_functions.cpp` is deleted rather than merely left unbuilt: nothing referenced it, and adding
  it to a CMake target would have meant new link plumbing for both firmware and JIT kernels for no
  benefit.
- `fds_go_blocking` and `fds_done_blocking` promised FIFO back-pressure their bodies did not
  implement, and nothing used either. Both are removed. The `ad_enable = true` path of `fds_go` and
  `fds_done` already spins on the FIFO-full register, which is the behaviour those comments
  described.

Five read accessors now exist. Three came from the bring-up tests and are used by `fds_poll` on both
sides: `fds_read_group_count` on the dispatch side, `fds_read_group_status` and `fds_read_de_status`
on the engine side. Two were added for the Gate 0 experiments: a dispatch-side `fds_read_group_status`
for the quiet-lane map, and `fds_read_neo_status` for the raw done inbox, which had been dropped when
nothing used it. The engine-side status accessor is documented as a live
per-lane mask rather than a latch, per open question 1.

*Side effect:* the header gained live consumers for the first time, so its defects stopped being
harmless — which is how the last two were found.

### 3. Completion ordering on every worker hart. **Correctness blocker until proven.**

The ROCC macros are `asm volatile` with no `"memory"` clobber, FDS bypasses NOC ordering, and the
per-kernel NOC completion checks in `dmk.cc:113-123` are disabled. A compiler barrier orders the
compiler only. `noc_async_full_barrier()` on DM0 is not yet sufficient evidence: current code
does not prove that it drains transactions issued by subordinate DMs, and its posted-write
condition is "sent", not necessarily committed at the remote destination.

**What reading the current path adds.** The NOC-atomic completion it replaces has no barrier before
it either: `dm.cc:410` calls `notify_dispatch_core_done` straight after `wait_subordinates()`, and on
Quasar that helper is a plain `noc_fast_atomic_increment` (`firmware_common.h:224`). What holds
today's path together is the kernel-level contract that a kernel drains its own NOC traffic before
returning — and the firmware checks that were meant to catch violations are commented out at
`dmk.cc:113-123`, "TODO enable once NOC is ready".

Two consequences. FDS introduces no new requirement in the ordinary case: a kernel that honours the
drain contract is as safe over the sideband as over the atomic. What FDS removes is the coincidental
backstop — when a kernel forgets to barrier, an undrained write and the atomic are both NOC traffic
and the atomic may still arrive behind the write, so the bug stays latent. Dedicated wires cannot be
ordered behind NOC traffic, so the same omission becomes reliable corruption.

That narrows this change from one open question to three tractable ones: does a DM0-level barrier
drain traffic issued by subordinate DMs, or must every hart drain before signalling subordinate
completion; is `noc_async_full_barrier`'s posted-write condition "sent" or "committed at the
destination"; and can the `dmk.cc` assertions be re-enabled so a non-draining kernel is caught rather
than inferred. The third is plain work and worth doing regardless of FDS.

**The hazard is demonstrated.** `DispatchEngineWriteOrdering` has a worker write 32 KB into the
dispatch core's L1 over the NOC and drive its done, and the dispatch engine read that payload the
moment the done appears, in two arms:

```
barrier=true:  tail word=0xdef1fff (expected 0xdef1fff)  mismatched words=0 of 8192
barrier=false: tail word=0xbaadf00d (expected 0xdef1fff)  mismatched words=16 of 8192, first at index 8176
```

Without a barrier the last 64 bytes of the transfer — one cache line — were still unwritten when the
done was observed. With a barrier the payload was intact. So a fence is **mandatory**, on this build,
with evidence rather than by argument, and `noc_async_write_barrier()` on the issuing hart is
sufficient for that hart's own writes. That answers the second of the three questions above and turns
the first into the only real unknown: subordinate-issued traffic.

Two cautions. The window is 0.2% of the transfer — at 4 KB it was invisible — so congestion and
silicon timing will change its size, and a clean result from a smaller test means nothing. And the
first version of this harness reported clean on both arms because it had work between the write and
the signal on one side and between the signal and the read on the other; a negative result here is
only as good as those gaps.

**FDS removes a safety net, but does not add a cost.** The control arm repeats the unbarriered case
with completion announced by `noc_semaphore_inc` on `NOC_UNICAST_WRITE_VC`, the mechanism and channel
the current path uses:

```
barrier=false signal=fds:         tail word=0xbaadf00d  mismatched 16 of 8192
barrier=false signal=noc-atomic:  tail word=0xdef1fff   mismatched  0 of 8192
```

The atomic is ordered behind the payload write; the sideband is not. **Both arms are
contract-violating cases, though, so read this narrowly.** Real kernels drain before returning —
`noc.async_write_barrier()` at the end of a writer is the norm, as in
`test_kernels/dataflow/l1_to_dram.cpp` and `writer_unary.cpp` — and `wait_subordinates()` means DM0
signals only after every subordinate kernel, and hence every subordinate's own barrier, has finished.
For conforming kernels the drain is therefore already paid today, and FDS adds no new cost. The
benefit case in "Context" stands: one atomic saved per worker per launch, against a barrier that is
already there.

What the comparison does show is that FDS removes a safety net. A kernel that fails to drain is
currently rescued by NOC ordering, because the atomic cannot overtake data it shares a road with. Over
the sideband the same omission becomes reliable corruption. That is a robustness requirement, not a
latency one, and it is sharpened by the firmware checks at `dmk.cc:113-123` being commented out, so
nothing catches a non-draining kernel today.

Where barriers actually live is asymmetric, which is worth recording. The **go** direction is covered
on the dispatcher side — `cq_dispatch.cpp:1094` under the wait command's `barrier` flag, and the
barrier before multicasting the launch message noted at `dm.cc:289`. The **done** direction has no
firmware barrier at all: `dm.cc:410` signals straight after `wait_subordinates()`. It rests entirely on
the kernel contract composing across harts, which is now the narrow question change 3 has to answer.

Two things this does not establish. Whether the NOC ordering is architectural or an artifact of this
model's timing — if the latter, today's path is equally exposed to a non-draining kernel. And whether
the atomic arm's clean result is ordering rather than the arm noticing its signal later, since it polls
an uncached L1 word where the FDS arm reads a coprocessor register. Recording each arm's poll-iteration
count would separate those.

Gate 0b must identify a hardware-supported remote-completion fence and prove whether its scope is
hart-wide or tile-wide. If the scope is per-hart, every enabled data-movement kernel hart drains
before signalling subordinate completion; DM0 then performs the compiler/FDS ordering step. Add
a congested test in which a subordinate DM performs the final write and the consumer validates
data immediately after FDS completion. If no fence can establish remote visibility, retain the
NOC completion transaction and abandon the claimed critical-path removal.

The required fence is a performance cost and needs an explicit before/after dispatch-latency
gate; replacing one atomic with a more expensive full drain is not automatically a win.

### 4. Worker transport selection and FDS output. **Medium confidence.**

Two acknowledgement sites remain under `hartid == 0`:
- normal program end, `dm.cc:385-406`, gated on `kernel_config.mode == DISPATCH_MODE_DEV`;
- the early path, `dm.cc:295-313`, for `RUN_MSG_RESET_READ_PTR` and `RUN_MSG_REPLAY_TRACE`,
  where **no kernel ran**. `RUN_MSG_RESET_READ_PTR_FROM_HOST` must stay non-acknowledging.

Add a persistent worker-visible `completion_transport_mode`, populated from the host's resolved
dispatch placement before worker reset. It selects:

- `NOC_ATOMIC` for Wormhole/Blackhole, `TT_METAL_TENSIX_DISPATCH_CORES=1`, and any Quasar device
  containing an unsupported completion producer such as active Ethernet;
- `FDS` only for the validated dispatch-engine/Tensix-only path.

`resolved_dispatch_core_type` is host state and cannot be referenced from `dm.cc`; this explicit
device field is required. Both normal and reset/replay acknowledgement sites branch on it.

For FDS, the group is
`(go_msg.dispatch_message_offset % DISPATCH_MAX_MESSAGE_ENTRIES) + 1`, never offset-plus-one.
On accepting every acknowledgement-producing GO/reset/replay signal, clear the worker output
with `fds_clear_done()` before work begins. At completion, execute the proven completion fence,
then a compiler barrier, keep `CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER()` in place, and
write the group ID. Init also defensively clears the output and explicitly disables worker-side
interrupts/auto-dispatch; no worker-side group configuration is needed because go remains NOC.

`notify_dispatch_core_done` retains its NOC-atomic body for all non-FDS modes. This is no longer a
`dm.cc`-only change: transport configuration plumbing and possibly per-hart kernel epilogues are
part of the correctness scope. Remove NOC-atomic accounting only on the FDS branch.

### 5. Tile-wide owner implementation. **Medium confidence pending Gate 0.**

Do not add independent FDS state to each `cq_dispatch_subordinate.cpp`. First reserve one
dispatch-engine processor as the tile-lifetime owner, or prove queue-selective hardware routing
that is equivalent. The owner exports named state for triage: active/pending masks and
generations, active CQ generations, epoch state, credited masks, raw status snapshots, and clear
request/ack generations.

The owner configures the **dispatch-side** filter, `GROUPID_ENABLE`, thresholds, interrupts, and
auto-dispatch state for every active group. Software masks do not replace hardware enable
programming. Configuration is read back before the owner reports ready.

Poll at the top of every service-loop iteration and inside every blocking path. If the owner
reuses dispatch_s control flow, the known sites are `wait_for_workers`, `cb_acquire_pages`,
go-permission synchronization, real-time-profiler flush, post-terminate
`cb_wait_all_pages`, and shutdown-sem waits. Preserve device-print/profiler hooks. A generic
blocking primitive used by the owner must accept a poll callback so future waits cannot silently
starve FDS.

Diagnostics compare raw status against the active mask before filtering and use bounded timeout
dumps for missing/extra/swapped bits. A duplicate-level assert is invalid because held credited
bits are expected until clear.

### 6. Termination and owner lifetime. **Medium confidence.**

dispatch_d terminate precedes dispatch_s terminate, but independent buffers allow dispatch_s to
stop first. Quasar also compiles out the existing `publish_dispatch_d_noc_count` path that sets
`dispatch_d_shutdown_sem_id`.

Add explicit Quasar shutdown bits, one per CQ. Each dispatch_d sets its bit only after its final
worker wait, clear acknowledgement, and NOC barriers. The tile owner keeps polling FDS and clear
requests while waiting for all configured CQ bits, including during page drain. It exits only
when every CQ is shut down and every epoch is quiescent. The service must not be tied to whichever
per-CQ terminate command arrives first.

### 7. Host/device protocol, ownership, and staged configuration. **Medium confidence.**

Extending `CQ_DISPATCH_SET_SUB_DEVICE_WORKER_COUNTS` is a **command ABI change**, even if its
command ID is retained. Its versioned payload carries an explicit word count, configuration
generation, worker counts, and one-hot-verified masks. Update calculators, both parsers, debug
decoding, cached command builders, and unit tests together; a word count alone cannot make an
old host/new kernel mismatch safe.

Do not derive FDS bit position from `functional_workers` list order. Gate 0 establishes physical
wire mapping; one architecture/HAL-owned function converts physical worker coordinates to bits
and rejects unknown or harvested mappings.

Reconfiguration is staged, not a call-order swap:

1. Send new counts/masks as **pending** generation B while active generation A remains in use.
2. The go/reset command waits for all A completions.
3. At that exact boundary, the owner clears and quiesces A, activates B atomically, and
   acknowledges the switch.
4. Only then send `RUN_MSG_RESET_READ_PTR` to B workers and collect acknowledgements with B.

CQ ownership transfer likewise carries `(sub_device, new_cq, generation)` to the tile owner and
waits for acknowledgement before host `CQOwnerState` permits the new queue to launch.

### 8. Interfaces preserved versus changed.

Preserve the public cumulative-count interpretation, L1 slot addresses, wait counts, and
Wormhole/Blackhole behavior. The host can keep emitting `WAIT_MEMORY`/`CLEAR_MEMORY`, but Quasar
`process_wait` must recognize completion-region clears and turn them into owner clear-generation
requests rather than direct stores.

Change explicitly:

- the worker-count command payload and ownership-transfer protocol;
- worker launch metadata for completion transport;
- Quasar completion-word clear handling;
- owner placement/lifetime and shutdown handshakes;
- telemetry to consume owner-published L1 counters;
- all synthetic/non-Tensix completion paths or the eligibility gate that excludes them.

The cached command mirrors must be audited because their wire flags may remain the same while
the Quasar device-side meaning changes.

---

## Can `WORKER_COMPLETION_SEMAPHORES` be deleted?

**No.** Five independent reasons, in order of how hard they are to remove:

1. Under this design it *is* the rendezvous — every consumer polls it.
2. Consuming an FDS done has exactly one consumer, so dispatch_d needs a handshake word from
   the tile owner regardless. Even the aggressive alternative I evaluated (delete the region,
   have the owner acknowledge dispatch_d through new wait flags) still needs a shared L1
   word per queue. "No shared L1 at all" is not reachable.
3. The Tensix-dispatch-core path keeps NOC atomics and needs the region intact.
4. A dispatch-engine tile running a user program keeps NOC atomics, so
   `DISPATCH_MESSAGE_ADDR` must stay valid and the region allocated.
5. Host command streams address it directly through `CLEAR_MEMORY`; on the FDS path the device
   converts that operation into an owner-mediated clear.

What *does* change on eligible FDS launches is its writer: the tile owner becomes the sole
publisher. The region must remain NOC-addressable because fallback, unsupported, and
non-Tensix paths can still use the existing transport; do not weaken its address contract.

I also evaluated the delete-it design in full. It needs two new wait flags, a doubled
command emission in `add_dispatch_wait` with a matching calculator split across roughly a dozen
call sites, a new dispatch_d↔dispatch_s ordering invariant to avoid deadlock, and it touches
shared Wormhole/Blackhole host code. It buys 256 bytes of L1. Not worth it.

---

## Testing strategy

**Gate 0 — FDS semantics and topology. Transport passes; the gate does not, yet.** Nothing beyond the
shim and additive bring-up tests lands until:

- the sideband carries a signal at all — **done**, on the 2026-08-20 simulator build, in both
  directions, with dones aggregating across two tiles;
- dispatch-side configuration is proven and read back — **done**;
- processor-window configuration and clear isolation are known — **done**, and the answer is that
  there is nothing to isolate: one block per tile, shared by every data-movement core on it;
- a go for one group does not release another group's workers — **done**:
  `DispatchEngineSubDeviceGroupIsolation` shows the unsignalled group's tile seeing the signalled
  value on a shared wire without latching it. Note what this means for the test: before the lanes
  carried anything, its two negative assertions passed vacuously, so a green result from it is only
  meaningful alongside a passing `DispatchEngineAllWorkers`;
- a done credits only its own group — *not tested*: an unsignalled group's workers never see a go, so
  they never drive done. Needs a worker driving done unprompted;
- every physical worker bit is mapped, including harvested layouts — *not done, now possible*. The
  next piece of work; see change 1 for the method and its two prerequisites;
- status/inbox deassertion and same-ID re-arm are repeatable — *not done, now possible*. Liveness is
  settled: status asserts, is live, and names its source instance;
- exclusive routing or the tile-wide-owner visibility model is established — *not done*. Routing
  across dispatch instances needs a descriptor with more than one dispatch tile.

The items marked done for the register interface were established by instrumentation that has since
been removed from the tests, as change 1 describes. The findings stand — they were measurements — but
the next experiments need that instrumentation restored.

**Gate 0b — completion ordering. Hazard demonstrated; scope open.**
`DispatchEngineWriteOrdering` covers the single-hart case and shows the race is real on this build and
that the issuing hart's barrier closes it. Still to add: heavy NOC congestion, a *subordinate* DM
issuing the last write, delayed delivery, a sweep of transaction types, and the NOC-atomic control arm
that says whether today's path is safe by contract or by accident.

Silicon or a certified model is still required before the fence's scope and cost become a contract.
What no longer needs silicon is the necessity: a fence is required, measured.

**Gate 1 — two workers, one dispatch engine. Partly done.** An earlier revision of this section cited
`quasar_simulation_2x3_arch_fast_dispatch.yaml` as having one usable worker. **No such file exists.**
The two shipped 2x3 core descriptors are `quasar_simulation_2x3_arch.yaml`, with two worker tiles and
no tensix dispatch cores, and `quasar_simulation_2x3_arch_tensix_dispatch.yaml`, with one worker tile
because a tile is given over to dispatch. The one-usable-worker property belongs to the second, which
is the interim `TT_METAL_TENSIX_DISPATCH_CORES` path that keeps NOC atomics and that FDS does not
target. So the original caveat pointed at a file that does not exist and, as far as can be told from
the descriptors, at a constraint that does not apply to the FDS path. Worth re-checking against
whoever wrote it before relying on either reading.

What has been run is `emu-quasar-2x3_DISPATCH` under slow dispatch, whose soc descriptor lists
`functional_workers: [0-1, 1-1]` and `dispatch: [1-2]`.

The multi-worker case now exists under slow dispatch as `DispatchEngineAllWorkers` in
`test_quasar_dispatch_engines.cpp`. It fans the handshake out to every worker tile the device
offers, one data-movement core per tile, and requires the dispatch engine to accumulate one done
per tile before its wait is satisfied. This is the configuration in which a wrong mask, a go aimed
at the wrong lane, or a done count that stops at the first arrival can fail rather than pass. Both
tests share one launch path, so they differ only in how many worker tiles take part. It skips itself
on a single-tile descriptor, where it would add nothing, and asserts rather than silently truncating
if a grid ever exceeds the 32 available done lanes.

It passed on the 2026-08-20 build with a done count of two, which is the measurement that put
transport beyond doubt — see "Gate 0 result".

Fast dispatch is still uncovered: every FDS test so far requires `TT_METAL_SLOW_DISPATCH_MODE`,
because kernels on dispatch-engine cores need it. Still to cover: staggered arrival in both orders,
many pipelined epochs with no intervening `Finish`, reset/replay acknowledgements, and owner clear
generations. Re-arm now passes for a single pair of epochs, so the pipelined cases are the next
escalation of it rather than blocked behind it.

**Gate 2 — two CQs on one dispatch-engine tile.** Enable the actual dispatch-engine two-CQ
fixture. Keep the inactive CQ's command stream idle so its kernel remains in the page-acquire
spin, then prove it never credits the active CQ's assertion. Transfer one sub-device between CQs
after odd and even numbers of epochs through both `Finish` and event ownership paths. Verify the
device ownership generation, destination L1 slot, and clear acknowledgement on every transfer.

**Gate 3 — existing coverage wired to the right path.** Add to
`tests/scripts/quasar/quasar_regression_tests.yaml` under `config: 2x3_DISPATCH`:

- `test_quasar_trace.cpp`, including reset/replay acknowledgements;
- `test_quasar_events.cpp`, including owner-mediated clear;
- a Quasar-enabled multi-sub-device suite covering group IDs 1 through 8;
- shutdown with dispatch_d blocked at each wait site.

Keep the existing `TT_METAL_TENSIX_DISPATCH_CORES=1` entries and assert they select
`NOC_ATOMIC`, not FDS.

**Negative and diagnostic tests:**

- missing mask bit → bounded timeout with raw-status and active-generation dump;
- extra/swapped bit → raw outside-mask or one-hot-mapping diagnostic;
- clear request racing the final worker credit → no lost clear and no resurrected count;
- stale auto-dispatch enabled before init → explicit disable/readback succeeds;
- unsupported active-Ethernet/synthetic producer → whole device selects NOC atomic, or the
  contribution is routed through the owner if that support is implemented;
- command payload version/word-count mismatch → deterministic device assert, never prefetch
  desynchronization;
- CQ1/sub-device7 → group 8 and L1 slot 15, never FDS id 0.

**Wormhole/Blackhole:** run standard fast-dispatch, trace, event, sub-device, and two-CQ suites.
The shared-code blast radius now includes a command ABI extension, transport metadata,
ownership commands, and Quasar-specialized clear handling; it is not limited to a neutral call
swap.

---

## Prerequisites and open questions

Confirm before writing dispatch code:

0. **Does the sideband carry a signal at all?** *Answered: yes*, on the 2026-08-20 simulator build,
   in both directions, with dones aggregating across two tiles. It did not on the previous build,
   which is what had blocked every question below that needs an assertion to arrive.
1. **Status semantics.** *Answered for assertion, open for deassertion.* `GROUPID_STATUS` is a live
   per-lane mask, read-only, not latched, not gated by `GROUPID_ENABLE`, with no read side effects
   observed, and it asserts under a real go and names the source dispatch instance. The status-mask
   model in the publish loop therefore stands. Deassertion has not been watched, and the raw inbox
   remains useful alongside status because a foreign group's value appears there without latching.
2. **P1 repeatability.** *Answered: yes*, on this build. A sink-side inbox clear holds against a live
   source, the go de-asserts and the group latch follows, and a second done for the same group
   produces exactly one new credit. The design's held-level assumptions and its `CLEARING` state are
   therefore implementable as written.
3. **Register topology.** *Answered.* Per **tile**, not per processor. One block serves every
   data-movement core on a tile, the decode is nine bits, and there is no banking. Independent
   enables per processor are therefore impossible, and cross-processor clears are unavoidable
   rather than something to test for — which is why the tile-wide owner is mandatory.
4. **Worker wire mapping.** *Partly answered.* Two tiles sit on lanes 0 and 4, one lane per tile,
   measured twice over by independent registers. The assignment *rule* is not established and
   harvested layouts are untried, so the HAL function must still reject topologies it has not been
   measured on. Do not infer from descriptor list order. If the stride of four reflects a per-engine
   lane space, a dispatch tile addresses at most eight workers.
5. **Completion fence.** Unanswered; belongs to Gate 0b. Narrowed by reading the current path — see
   change 3 — to three questions rather than one, with a harness now written for the first two.
6. **Routing.** Partly answered: dispatch instance 0 drove every go observed, read off the
   engine-side status field. The three-instance question still cannot be exercised, because only one
   dispatch tile exists in the simulator descriptor.
7. **ROCC availability.** *Answered.* Data-movement cores only. The Tensix engine processors
   cannot reach the interface, which is why "a raw `.word` compiling proves nothing" was the right
   caution: it compiles and executes on an engine processor and does nothing at all.
8. **Deglitch and reset defaults.** *Partly answered.* Reset values are zero for the filter, the
   auto-dispatch enable and the interrupt enable, and explicit disable and readback both work. The
   working handshake runs with the filter at zero; other settings are untested against a real
   assertion. Persistence across device reset is untested, and matters more now that held levels can
   actually be asserted across one.
9. **Owner placement.** Unanswered, but now constrained: the owner must be a data-movement core.
10. **Unsupported producers.** Decide whether active Ethernet and virtual-worker credits are
    routed through the owner or force device-wide NOC-atomic mode.

No authoritative FDS hardware documentation was available during review. Simulator behaviour must be
checked against RTL/silicon ownership, ordering, and clear semantics before it becomes a contract. The
lanes carrying on one simulator build is a starting point for that check, not a substitute for it.

### Go/no-go conditions

Do not proceed with FDS completion if any of these remains true:

- the sideband carries no signal between dispatch-engine and Tensix tiles on any available hardware
  or simulator configuration — *no longer true as of the 2026-08-20 build*;
- same-ID P1 cannot complete repeatedly without ambiguity — *cleared on this build*;
- no fence orders every issuer's remote writes before sideband done — *partly cleared*: the issuing
  hart's own barrier is proven sufficient for its own writes; subordinate-issued traffic is untested;
- two CQs cannot share one tile-wide owner or obtain proven exclusive routing;
- the worker-bit mapping is unknown for supported harvested layouts — *still true*: two tiles are
  mapped, no rule is established, and harvested layouts are untried;
- unsupported completion producers can still write the public L1 word concurrently;
- the required fence and owner polling erase the expected latency benefit — the fence is not itself a
  new cost, since conforming kernels already drain, so this rests on owner polling and on whatever the
  scope question in change 3 forces.

## Overall confidence

- **The register interface itself — high, and now measured rather than assumed.** Address decode,
  field widths, per-processor topology, read-only status, reset defaults, and which processors can
  reach it are all established.
- **FDS lane transport — high, on one simulator build.** Both directions carry and dones aggregate
  across tiles. The physical premise of the design is verified where it was previously unverified.
  What is not established is that silicon behaves the same way, and one build is one data point.
- **Persistent L1 interface and retaining the region — high.** Existing consumers still need it.
- **FDS as the physical rendezvous — medium-high as a design, partly proven as a mechanism.** The
  workload shape fits, status behaviour under assertion is now measured, and go-direction group
  isolation holds. Clear semantics, re-arm and ordering remain unestablished.
- **Tile-wide owner architecture — medium-high conceptually, medium for integration.** It closes
  the duplicate-reader race but requires new placement, ownership, clear, and lifetime plumbing.
- **P1 register protocol — medium-high on this build, from a passing two-epoch test.** Sink-side clear
  holds against a live source, the go de-asserts, and a second same-group done yields exactly one
  credit. What is not established is that this survives more workers, more groups, pipelined epochs
  without an intervening drain, or silicon.
- **Completion ordering — the risk is now measured rather than suspected.** Without a barrier a
  worker's last 64 bytes were demonstrably unwritten when its done was observed; with one they were
  not. High confidence that a fence is required and that the issuing hart's barrier suffices for its
  own traffic. Low confidence on scope: subordinate-issued writes, congestion, and cost are untested,
  and silent stale data remains the dominant failure mode if any of those is wrong.
- **Two-CQ and reconfiguration correctness — medium-low until device ownership generations and
  staged masks pass Gate 2.**
- **Performance benefit — unknown, and no worse than first thought.** Conforming kernels already
  barrier before returning, so FDS does not add a drain; the trade is one atomic saved against owner
  polling replacing an atomic's instantaneous visibility. Measure it on a rough prototype rather than
  after the full protocol exists.

Promotion to implementation requires explicit evidence for every go/no-go condition above. Three are
still open: the completion fence, the worker-bit mapping rule, and the two-CQ ownership model.

The physical premise is now established rather than assumed. Transport works in both directions,
dones aggregate, groups are isolated in both directions, and consecutive epochs of one group are
distinguishable — which together mean the owner state machine and the group-per-sub-device mapping
can be written as designed rather than redesigned around a hardware limitation.

The next steps split three ways. The lane-assignment rule needs a descriptor with more worker tiles,
or the specification; the test that measures it needs no changes. Gate 0b's central answer is in: ordering
matters, and it is the kernel drain contract — already honoured by conforming kernels — that satisfies
it, not anything new. What FDS changes is that a violation of that contract stops being hidden. So the
remaining work there is narrow: confirm the per-hart barriers compose to cover every hart before DM0
signals, and re-enable the assertions at `dmk.cc:113-123` so a violation fails loudly. Everything else — the transport-mode plumbing in change 4, the owner's placement and
lifetime in changes 5 and 6, the staged command protocol in change 7 — is ordinary implementation work
whose design questions are settled.
