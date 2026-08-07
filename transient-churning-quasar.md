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

This plan is now explicitly **conditional**. No dispatch implementation starts until Gate 0
settles the FDS register semantics and Gate 0b identifies a completion fence that orders every
issuing worker hart's writes before sideband done. A passing simulator test is necessary but is
not evidence that silicon has the same semantics.

**Gate 0 has now been run and does not pass.** The register interface is real and fully
characterised, but no signal crosses between the dispatch-engine tile and the Tensix tiles on the
available simulator configuration. See "Gate 0 result" below. Nothing in this plan proceeds until
that is resolved, and it is not resolvable by further software probing.

## Decision summary

| Question | Answer |
|---|---|
| What does FDS replace? | Tensix worker→dispatch transport on eligible launches. The cumulative 32-bit counting model remains the public interface. |
| Which hardware register drives it? | **Still undecided.** Gate 0 showed `GROUPID_STATUS` is a live, read-only, per-lane mask that is not gated by the enable register, which suits the design — but only idle lanes have ever been observed, so its behaviour under a real assertion is untested. |
| Which processors can talk to FDS? | **Data-movement cores only.** Gate 0 established that the Tensix engine processors cannot reach the interface at all, so the tile-wide owner must be a data-movement core and worker completion cannot be signalled from a compute kernel. |
| Who talks to FDS? | Exactly one **tile-wide** owner for all CQs on a dispatch-engine tile. A per-CQ `dispatch_s` owner is invalid unless hardware proves queue-selective isolation and routing. |
| Who writes the cumulative L1 words? | The tile-wide owner only. Clears, virtual-worker credits, and any non-FDS contributions must be serialized through it. |
| Can `WORKER_COMPLETION_SEMAPHORES` be deleted? | **No.** It remains the persistent interface consumed by dispatch waits and host-generated commands. |
| Gate | A worker-visible transport mode selected only when the resolved dispatch core is `CoreType::DISPATCH` and every completion producer is supported. `ARCH_QUASAR` is insufficient. |
| P2 fallback? | **No automatic fallback.** If P1 cannot be made repeatable, stop and redesign an explicit shared epoch protocol rather than silently reducing the sub-device limit. |

### Review disposition

The direction — convert ephemeral FDS state into persistent L1 counters — survives review. The
original per-CQ owner, plain RMW publication, call-order swap, P2 fallback, and `dm.cc`-only
scope do not. They are replaced below with a tile-wide owner, owner-mediated clears, staged
configuration, a worker-visible transport gate, and a fully tested same-ID epoch protocol.

---

## Gate 0 result: the register interface is proven, the sideband is not

Gate 0 has been run against the `quasar_simulation_2x3_arch` slow-dispatch configuration using
`tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp` and its dispatch-engine, data-movement
and Tensix-engine kernels. The outcome divides cleanly in two.

The register interface is real, complete, and now characterised in detail. **No signal has ever
crossed between the dispatch-engine tile and a Tensix tile, in either direction, from any
processor software can reach.** Gate 0 is blocked, and it is blocked on hardware configuration
rather than on anything further a software probe can settle.

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

The dispatch engine held group 1 in its outbox for entire runs, with the value read back. Twelve
data-movement blocks across both worker tiles drove done and held it. Throughout:

- the dispatch engine's quiet-lane map never changed from all thirty-two;
- all thirty-two of its raw inbox registers stayed at zero, and its group count at zero;
- every worker block's three raw inbox registers stayed at zero, and its group status at zero.

The raw inbox registers sit before all aggregation, so no group, enable or threshold setting can
account for them. Both sides' receive logic is demonstrably instantiated and evaluating its lanes
— that is what the quiet-lane maps mean — and both sides' transmit registers hold what is written
to them. Nothing passes between the two.

### What has been eliminated

| Candidate cause | How it was eliminated |
|---|---|
| Timing races, lost edges | Both signals are held rather than pulsed, with tens of seconds of simulated overlap |
| Count thresholds | Set to one, and the raw inboxes bypass counting entirely |
| Enable masks, group id choice | The raw inboxes sit before aggregation |
| Deglitch filter | Swept 0, 1, 2, 8 and 64 on both sides against held signals |
| Stale auto-dispatch or interrupt state | Both read as zero before being explicitly zeroed |
| Wrong processor | All twelve user data-movement cores across both worker tiles |
| Instrument error | Field-width truncation and cross-address probes prove real registers and correct addressing |

### What to put to the hardware configuration owners

1. Does the 2x3 simulation configuration instantiate and connect the fast dispatch signal lanes
   between the dispatch tile at `1-2` and the Tensix tiles at `0-1` and `1-1`?
2. If not, is there a configuration that does?
3. If it does, is there a block-level enable outside the two register maps? Neither map contains
   anything resembling one.
4. Which processor is the intended lane endpoint on each side?

### Confidence in these findings

High on the register interface and on the reachability finding. Each rests on a measurement shaped
so that the alternatives produce visibly different answers, and each was reproduced across
processors and tiles. High that no reachable processor can move a lane on this build.

The one possibility software cannot exclude is a block-level enable in a register we do not know
exists. Nothing in either map resembles one, but a register's absence from a description file is
not proof of its absence from the hardware.

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

---

## Changes, with confidence

Confidence is about *my analysis being right*, not about the change being easy.

### 1. Bring-up tests first — before any dispatch code. **Done, and blocking.**

This exists as `tests/tt_metal/tt_metal/test_quasar_dispatch_engines.cpp` under slow dispatch,
with three kernels: a dispatch-engine sender, a data-movement worker that runs on every user
data-movement core, and a Tensix-engine variant. The dispatch-engine kernel programs and reads
back the dispatch-side filter length, `GROUPID_ENABLE`, threshold, interrupt enable and
auto-dispatch enable, and the two directions are deliberately not chained: the worker drives done
part way through its wait whether or not a go arrived, so one run reports on each direction
independently.

Of the original matrix, these are settled:

- explicit disable and readback of interrupts and auto-dispatch — both read zero before being
  zeroed;
- read visibility from every candidate owner processor — all of them share one block per tile, so
  every processor sees the same registers and there is no isolation to establish;
- which processors can reach the interface at all.

These cannot be attempted until a signal crosses:

- all 32 worker inboxes one-hot, and harvested layouts;
- live versus latched `GROUPID_STATUS` under an assertion, and `GROUPID_COUNT`;
- the same-ID two-epoch P1 cycle;
- routing to every dispatch instance, and two processors polling concurrently.

Gate 0 was to choose status-mask versus inbox polling and decide whether P1 exists. It has done
neither, because both questions require an assertion to arrive. It has instead established that
none does.

*Side effect:* none — additive.

### 2. Make the vendored shim usable. **High confidence.**

- `overlay/fds_functions.hpp:10-11` — change the two bare includes to
  `"meta/fds_registers/..."`. This resolves by the quoted-include rule, so **no change to
  `qa_hal.cpp` `includes()` is needed**, which is the cheaper of the two fixes.
- `fds_functions.cpp:44-46` — `fds_clear_neo_status` uses `_REG_OFFSET` (0x004) where every other
  dispatch-side helper uses `_REG_ADDR` (0x204). **This is not a defect, contrary to the earlier
  reading here.** Gate 0 established that only nine address bits are decoded, so bit nine is
  discarded and both forms reach the same register. Making it consistent is still worth doing for
  readability, but nothing depends on it and it was never writing into the worker-side map.
- Both `fds_config_auto_dispatch(false, ...)` implementations currently skip the enable-register
  write. Change them to write zero unconditionally and read back zero in Gate 0; stale emulator
  state is one of the conditions this initialization is intended to survive.
- The helpers we need are one-line register accesses. Make them `inline` in the header and
  leave `fds_functions.cpp` unbuilt — adding it to a CMake target means new link plumbing for
  both firmware and JIT kernels for no benefit.
- Drop or fix `fds_go_blocking`/`fds_done_blocking` (`:41`, `:92`), whose comments promise FIFO
  back-pressure their bodies do not implement. We use neither.

*Side effect:* the header gains live consumers for the first time, so its defects stop being
harmless.

### 3. Completion ordering on every worker hart. **Correctness blocker until proven.**

The ROCC macros are `asm volatile` with no `"memory"` clobber, FDS bypasses NOC ordering, and the
per-kernel NOC completion checks in `dmk.cc:109-119` are disabled. A compiler barrier orders the
compiler only. `noc_async_full_barrier()` on DM0 is not yet sufficient evidence: current code
does not prove that it drains transactions issued by subordinate DMs, and its posted-write
condition is "sent", not necessarily committed at the remote destination.

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

**Gate 0 — FDS semantics and topology. Run, and not passed.** Nothing beyond the shim and additive
bring-up test lands until:

- every physical worker bit is mapped, including harvested layouts — *blocked, no assertion ever
  arrives*;
- dispatch-side configuration is proven and read back — **done**;
- status/inbox liveness, deassertion, and same-ID re-arm are repeatable — *blocked*;
- processor-window configuration and clear isolation are known — **done**, and the answer is that
  there is nothing to isolate: one block per tile, shared by every data-movement core on it;
- exclusive routing or the tile-wide-owner visibility model is established — *blocked*.

The prerequisite for the blocked items is a hardware or simulator configuration in which the lanes
carry a signal at all. See "Gate 0 result" above.

**Gate 0b — completion ordering.** Under heavy NOC congestion, make a subordinate DM issue the
last data write, delay its delivery, then signal completion. Validate the destination
immediately when the owner publishes done. Sweep issuing harts and transaction types. This gate
must run on silicon or a model explicitly certified for NOC/FDS ordering.

**Gate 1 — two workers, one dispatch engine.** The shipped
`quasar_simulation_2x3_arch_fast_dispatch.yaml` has one usable worker, so wrong masks and epoch
logic degenerate into success. The slow-dispatch descriptor is better than assumed: the simulator
soc descriptor lists `functional_workers: [0-1, 1-1]` and `dispatch: [1-2]`, and the bring-up test
now runs kernels on both worker tiles. A two-worker *fast-dispatch* descriptor is still needed.
Cover staggered arrival in both orders, many pipelined epochs with no intervening `Finish`,
reset/replay acknowledgements, and owner clear generations.

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

0. **Does the sideband carry a signal at all?** *New, and now the first question.* On the
   available simulator configuration it does not, in either direction, from any reachable
   processor. Every question below that needs an assertion to arrive is blocked behind this one.
1. **Status semantics.** *Partly answered.* `GROUPID_STATUS` is a live per-lane mask, read-only,
   not latched, not gated by `GROUPID_ENABLE`, with no read side effects observed. Only idle lanes
   have ever been seen, so behaviour under a real assertion is still unknown, and the choice
   between status-mask and raw inbox polling remains open.
2. **P1 repeatability.** Unanswered — requires an assertion.
3. **Register topology.** *Answered.* Per **tile**, not per processor. One block serves every
   data-movement core on a tile, the decode is nine bits, and there is no banking. Independent
   enables per processor are therefore impossible, and cross-processor clears are unavoidable
   rather than something to test for — which is why the tile-wide owner is mandatory.
4. **Worker wire mapping.** Unanswered — requires an assertion. Do not infer from descriptor list
   order.
5. **Completion fence.** Unanswered; belongs to Gate 0b.
6. **Routing.** Unanswered — requires an assertion. Note only one dispatch tile exists in the
   simulator descriptor, so the three-instance question cannot be exercised there.
7. **ROCC availability.** *Answered.* Data-movement cores only. The Tensix engine processors
   cannot reach the interface, which is why "a raw `.word` compiling proves nothing" was the right
   caution: it compiles and executes on an engine processor and does nothing at all.
8. **Deglitch and reset defaults.** *Partly answered.* Reset values are zero for the filter, the
   auto-dispatch enable and the interrupt enable, and explicit disable and readback both work. The
   filter has no observable effect anywhere between 0 and 64, though with no assertion to filter
   that is weak evidence. Persistence across device reset is untested.
9. **Owner placement.** Unanswered, but now constrained: the owner must be a data-movement core.
10. **Unsupported producers.** Decide whether active Ethernet and virtual-worker credits are
    routed through the owner or force device-wide NOC-atomic mode.

No authoritative FDS hardware documentation was available during review. Simulator behavior
must be checked against RTL/silicon ownership, ordering, and clear semantics before it becomes a
contract — and on the current configuration the simulator provides no lane behaviour to check.

### Go/no-go conditions

Do not proceed with FDS completion if any of these remains true:

- the sideband carries no signal between dispatch-engine and Tensix tiles on any available
  hardware or simulator configuration — *currently true, and it subsumes every condition below
  that depends on an assertion arriving*;
- same-ID P1 cannot complete repeatedly without ambiguity;
- no fence orders every issuer's remote writes before sideband done;
- two CQs cannot share one tile-wide owner or obtain proven exclusive routing;
- the worker-bit mapping is unknown for supported harvested layouts;
- unsupported completion producers can still write the public L1 word concurrently;
- the required fence and owner polling erase the expected latency benefit.

## Overall confidence

- **The register interface itself — high, and now measured rather than assumed.** Address decode,
  field widths, per-processor topology, read-only status, reset defaults, and which processors can
  reach it are all established.
- **FDS lane transport — currently absent.** Nothing crosses between tiles on the available
  configuration. Until that changes, the physical premise of this design is unverified, and no
  amount of software work advances it.
- **Persistent L1 interface and retaining the region — high.** Existing consumers still need it.
- **FDS as the physical rendezvous — medium as a design, unproven as a mechanism.** The workload
  shape fits, but status behaviour under assertion, clear semantics, and ordering are all still
  unestablished, and now demonstrably cannot be established on this build.
- **Tile-wide owner architecture — medium-high conceptually, medium for integration.** It closes
  the duplicate-reader race but requires new placement, ownership, clear, and lifetime plumbing.
- **P1 register protocol — low, and no longer "until Gate 0".** Gate 0 has run without producing
  any evidence either way, because the protocol needs an assertion to exercise. The available
  helper contract points to mandatory worker-side clear, and sink-side re-arm remains unknown.
- **Completion ordering — low until Gate 0b.** Silent stale-data corruption is the dominant risk.
- **Two-CQ and reconfiguration correctness — medium-low until device ownership generations and
  staged masks pass Gate 2.**
- **Performance benefit — unknown.** Measure after the required fence and owner polling exist.

The plan is ready for hardware bring-up work, not dispatch implementation. Promotion to
implementation requires explicit evidence for every go/no-go condition above.

The bring-up work has now been done as far as software can take it. The next step is not a code
change: it is establishing with the hardware configuration owners whether these lanes are
connected at all, and on which configuration. Until that returns an answer, this plan is parked
rather than merely conditional.
