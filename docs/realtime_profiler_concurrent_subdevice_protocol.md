# Blackhole Concurrent Sub-Device Profiler Protocol

## Status and scope

This is the Milestone 0 decision record for
`REALTIME_PROFILER_CONCURRENT_SUBDEVICE_REDESIGN_PLAN.md`.

The protocol applies only to the Blackhole, single-command-queue,
worker-dispatch path. For this feature the profiler eligibility gate rejects
Wormhole and Quasar rather than preserving the old single-stream profiler on
those architectures. It does not define model integration, operation core
allocation, multi-CQ behavior, trace replay, or remote-chip transport.

## Current-path evidence

The current branch contains the concurrent-profiler changes from
`20e839e1ff9032573690c031751ad58b5e967fed`. Inspection identifies two waits
that make that implementation unsuitable for asynchronous profiling:

1. `finish_realtime_profiled_program()` in
   `cq_dispatch_subordinate.cpp` waits until TRISC0's sampled stream count is
   exactly equal to the command target. This is profiler-only backpressure on
   dispatch and exact equality can be missed if the sampled counter advances
   past the target.
2. `drain_realtime_profiler_records()` waits for the reserved profiler BRISC to
   acknowledge every record. It is called on local-queue pressure, profiler
   flush, and termination. This couples dispatch progress to BRISC, NCRISC, D2H
   socket, and host progress.

There are two more downstream waits. They do not currently execute directly on
the application dispatch RISC, but they must not be allowed to propagate
backpressure into it:

- `realtime_profiler_read_and_enqueue()` waits while the reserved-core L1 ring
  is full.
- `push_entries_to_host()` waits in the D2H socket reserve path when host FIFO
  capacity is unavailable.

The authoritative Blackhole completion signal is the dispatch stream's
`STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX` value. The dispatch_s TRISC0
kernel already samples this register on the same physical dispatch core as the
dispatch_s NCRISC and captures the device wall clock on a change. The current
implementation writes only the latest timestamp and count per stream, which can
be overwritten before dispatch_s consumes them.

The current go-signal command does **not** carry enough information to calculate
the completion target for a partitioned sub-device. Its compile-time
`num_worker_cores_to_mcast` value is the whole physical compute grid, while only
the selected sub-device's workers increment that sub-device stream.

The host already calculates the exact per-launch completion contribution as
`num_workers` in `FDMeshCommandQueue::enqueue_mesh_workload()`. Milestone 1 will
carry that value as an eight-bit `profiler_num_workers` field in
`CQDispatchGoSignalMcastCmd`. Blackhole's maximum Tensix plus virtual ETH worker
count normally fits in eight bits. A larger count is encoded as zero so the
application still runs, emits one host warning per process, and is attributed
to `unsupported_launch_drop_count` rather than silently folded into another
loss category.

The command remains 16 bytes by narrowing `wait_stream` from 16 bits to 8 bits;
the dispatch stream index is bounded well below 256 by the hardware stream
count. This avoids doubling every go-signal command to 32 bytes. The field is
patched together with `profiler_program_id` for ordinary and traced command
sequences, but the new concurrent profiler does not claim trace-replay support.

The target for a profiled launch is therefore:

```text
target = wait_count
       + profiler_num_workers
```

using the stream counter's existing wrap semantics.

The device stream counter is a `MEM_WORD_ADDR_WIDTH == 17` field. Descriptor and
watermark targets are masked to this 17-bit domain, and TRISC0 uses the existing
half-range modular greater-than-or-equal comparison. A natural transition from
`0x1ffff` to `0` is forward progress; it does not reset profiler state or discard
any descriptor or watermark. Each per-launch advance is at most 255, far below
the half-range ambiguity boundary.

The virtualized-ETH workaround injects synthetic completion deltas into
`first_stream_used`, which can satisfy an unrelated first-stream descriptor
early. Concurrent interval publication is therefore excluded from dispatch_s
images built with virtualized-unicast support. Such launches still run normally
and increment `unsupported_launch_drop_count`; the dispatch workaround itself
is unchanged.

TRISC0 and dispatch_s NCRISC share the dispatch core's L1. This permits a local,
bounded producer/consumer protocol without adding NOC traffic to the program
start or completion critical paths. The reserved profiler BRISC remains the
only producer of its existing BRISC-to-NCRISC L1 ring.

## Baseline evidence

Hardware used for Milestone 0 is the local four-chip Blackhole P150b QuietBox
(`bh-qb-12-special-pjosipovic-for-reservation-62599`), firmware bundle
19.10.0, KMD 2.10.0, IOMMU enabled, and 800 MHz AI clock.

After rebuilding `unit_tests_dispatch` to eliminate a stale test/library ABI,
the current branch passes all five existing Blackhole sanity tests:

```text
RealtimeProfilerSanity.FiveProgramsBackToBack
RealtimeProfilerSanity.CloseDrainsRegisteredCallback
RealtimeProfilerSanity.ThrowingCallbackIsIsolated
RealtimeProfilerSanity.LastProgramRecordDeliveredOnFinish
RealtimeProfilerSanity.TraceReplayResolvesKernelSources
```

A one-second `RealtimeProfilerStress.PeakLoadPreservesRecords` attempt did not
reach its replay summary after more than eight minutes. A debugger snapshot
showed the host blocked in `FDMeshCommandQueue::finish_nolock()` waiting for the
device completion event. The run was stopped locally. It is retained as failure
evidence for the current blocking path, not treated as a throughput baseline.

Current static footprint from the generated Blackhole kernels is:

| Component | Current maximum measured image payload |
| --- | ---: |
| dispatch_s NCRISC | 7,848 B text, 80 B data, 452 B BSS |
| dispatch_s TRISC0 monitor | 1,232 B text |
| reserved profiler BRISC | 2,704 B text, 24 B data |
| reserved profiler NCRISC | 3,336 B text, 96 B data, 40 B BSS |
| dispatch_s profiler message | 4,424 B L1 |
| reserved profiler-core ring and socket config | 262,336 B L1 |

These are the largest payloads observed across the Milestone 0 configurations,
not one same-configuration image set. They are regression anchors rather than
the operands used for the component deltas below. Milestone 1 reports a matched
before/after image pair for its gate arithmetic.

## Selected protocol

### Ownership

| State | Producer | Consumer |
| --- | --- | --- |
| Per-stream start descriptor ring | dispatch_s NCRISC | dispatch_s TRISC0 |
| Completed interval ring | dispatch_s TRISC0 | dispatch_s NCRISC |
| Single transport mailbox B | dispatch_s NCRISC | reserved profiler BRISC |
| Reserved-core L1 ring | reserved profiler BRISC | reserved profiler NCRISC |
| D2H pages | reserved profiler NCRISC | host receiver |

Every mutable queue has exactly one producer and one consumer. No atomic
read-modify-write operation or lock is required.

### Per-stream start descriptors

Each monitored stream receives a power-of-two ring of four descriptors. A
descriptor contains:

- runtime ID;
- start timestamp high and low words;
- stream completion target;
- reset generation.

dispatch_s captures the start tick and publishes the descriptor immediately
before sending the go signal. Publishing consists of ordinary L1 stores, a
Blackhole `fence w,w`, and a final producer-index store.

If the descriptor ring is full, dispatch_s increments
`start_descriptor_drop_count` and launches the program without profiling it.
It never waits for TRISC0.

TRISC0 does not impose a device-cycle deadline on an unmet descriptor because a
valid program may run arbitrarily long. If an unmet head blocks a full ring, it
increments `stuck_descriptor_head_count` once for that blocked episode. This
makes a bad target diagnosable without dropping a valid long-running interval.

Each descriptor also carries the stream reset generation current at publication.

### Completion correlation

TRISC0 continuously samples all configured stream counters. For each stream it
examines the descriptor at the consumer index and applies the dispatch stream's
wrap-aware greater-than-or-equal comparison to the descriptor target.

When one descriptor target is satisfied, TRISC0 captures the end tick and emits
one interval. If more than one descriptor is already satisfied in the same
sample, only the newest satisfied descriptor can receive an accurate end tick;
older satisfied descriptors are discarded and counted in
`completion_observer_drop_count`. They must not be emitted with an invented or
duplicated completion timestamp.

TRISC0 advances the descriptor consumer index whether the interval is emitted
or counted as lost. It never waits for downstream capacity.

### Completed interval ring

TRISC0 publishes completed intervals into the existing 128-entry dispatch-core
record ring. Each entry remains eight words so the existing mailbox and D2H record
format can be retained:

```text
word 0  start timestamp high
word 1  start timestamp low
word 2  runtime ID
word 3  schema/type/stream metadata
word 4  end timestamp high
word 5  end timestamp low
word 6  successful interval sequence
word 7  cumulative source-drop snapshot
```

The successful interval sequence advances only when a complete record is
accepted into this ring. If the ring is full, TRISC0 increments
`completed_record_drop_count`, consumes the descriptor, and continues.

TRISC0 stores the record words, executes `fence w,w`, and publishes the producer
index last. dispatch_s uses the two-step Blackhole cache invalidation described
below before reading the producer index and record payload.

### Bounded dispatch forwarding

dispatch_s replaces `drain_realtime_profiler_records()` with
`service_realtime_profiler_once()`:

1. If mailbox B is not idle, return immediately.
2. If the completed interval ring is empty, return.
3. Copy one completed record into mailbox B.
4. Signal the reserved profiler BRISC with the existing inline NOC dword write.
5. Advance the completed-ring consumer index and return.

There is no acknowledgement loop. At most one record is forwarded per call.
The service function is invoked at bounded progress points already visited by
dispatch_s:

- once per command-loop iteration;
- while waiting for command-buffer input;
- while waiting for dispatch_d permission or worker completion;
- after processing a profiler flush request.

This keeps the common action to a few local loads and branches. A NOC signal is
issued only when a record is ready and mailbox B is idle.

Blackhole has no distinct uncached L1 alias: `uncached_l1_ptr()` is an identity
operation on this architecture. Both cross-RISC consumers therefore use an
explicit cache-invalidation sequence. Before reading a producer index, the
consumer calls `invalidate_l1_cache()`. After observing a non-empty queue and
before reading the reusable payload slot, it calls `invalidate_l1_cache()`
again. A fresh producer index must never be paired with a payload line retained
from an earlier use of that ring slot.

This rule applies to TRISC0 consuming NCRISC-published start descriptors and to
dispatch_s NCRISC consuming TRISC0-published completed records. It is required
regardless of the consumer's initial cache state; the implementation must not
depend on TRISC0 inheriting a disabled or empty data cache.

### Reserved profiler core

The BRISC retains the existing mailbox-B NOC-read protocol and remains the sole
producer of the reserved-core L1 ring. Its full-ring behavior changes:

- interval record: increment `transport_drop_count`, acknowledge mailbox B
  mailbox, and continue;
- clock-sync marker: retain the existing explicit sync behavior; sync does not
  execute on the application dispatch path.

The BRISC must not wait for reserved-ring capacity while handling an interval
from dispatch_s. The NCRISC and D2H socket can stall independently without
stalling program dispatch; pressure becomes counted interval loss.

## Milestone 2 watermark protocol

### Device request

One explicit exact collection allocates a monotonically increasing 32-bit batch
watermark ID, skipping reserved value zero, and registers the participating
sub-device stream mask with the profiler manager. Every
`CQ_DISPATCH_CMD_RT_PROFILER_FLUSH` emitted by that Finish carries the same batch
ID. A normal Finish carries watermark ID zero, which bypasses both collection
registration and the device watermark path.

After its existing worker wait completes, dispatch_s publishes a per-stream
watermark request containing:

- watermark ID;
- stream completion target from the flush command;
- a request generation.

This is a dedicated per-stream slot, not part of the start descriptor ring, so
a full descriptor ring cannot lose the request. A second request reaching an
occupied slot is counted as watermark-control loss and never stalls dispatch.

### Device completion

TRISC0 marks a per-stream watermark ready only after:

- the stream counter reached the requested target; and
- every start descriptor on that stream whose target is at or before the
  requested target has been emitted or counted as dropped.

The descriptor scan consumes every satisfied descriptor before checking the
watermark target. An unmet descriptor is necessarily for a later target and
does not delay the watermark.

Watermark generations are directional. If a request carries a generation newer
than TRISC0 has adopted, TRISC0 waits for its next pass to adopt that reset epoch
before comparing the target. If the request is older than the adopted epoch,
the old counter target is no longer observable; TRISC0 completes it with the
protocol-error marker because dispatch reset already quiesced the prior epoch.

The ready state snapshots:

- successful interval sequence;
- start-descriptor drops;
- completion-observer drops;
- completed-record drops;
- completed-record producer index that must be forwarded first.

dispatch_s forwards a watermark only after its completed-record consumer index
has reached the snapshotted producer index. Thus the watermark cannot overtake a
successfully accepted interval on the dispatch core.

The watermark uses the existing eight-word transport payload with a reserved
record type. Its payload is watermark ID, successful interval sequence,
normal or protocol-error marker ID, schema/type/stream metadata, cumulative descriptor loss,
cumulative observer loss, cumulative record loss, and cumulative transport
loss. Interval publication leaves one reserved slot in the profiler-core ring,
and the reserved profiler BRISC adds the transport-loss snapshot before
enqueueing the control page. If even that control slot is temporarily occupied,
BRISC retains the mailbox notification until NCRISC makes space; dispatch_s
does not wait for that acknowledgement. If termination occurs while the ring is
completely full, BRISC counts the occupied mailbox as transport loss before it
acknowledges and exits.

### Host completion

The host receiver consumes watermark pages internally and does not publish them
as program callbacks. Per device and batch ID, it stores the set of participating
streams observed plus their cumulative counter snapshots, then notifies
collection waiters.

The host collection result contains:

- requested watermark;
- observed participating-stream mask per active device;
- records received since the caller's baseline snapshot;
- descriptor, completion-observer, completed-record, aggregate source, and
  transport drop deltas;
- host callback-ring drop information remains callback-specific and is not
  conflated with device loss;
- timeout and protocol-error state.

A batch is complete only when every registered participating stream is observed
for every active device. A later batch does not satisfy an earlier batch, because
streams can become ready out of order. Ring emptiness is never used as proof of
completion.

`experimental::FinishAndCollectProgramRealtimeProfiler()` is the exact-batch
API. It registers a retained batch while holding the command-queue API lock,
executes the normal Finish path, releases that lock, and waits for that batch ID.
The timeout is host control-plane policy only; interval duration fields remain
raw device start/end ticks. Normal `Finish` creates no batch and never waits for
host profiler progress.

The interval parser now preserves schema version, record type, command-queue
identity, dispatch stream, and successful-publication sequence in
`ProgramRealtimeRecord`. Command-queue identity is fixed at zero because this
protocol supports one command queue only. Collection results report expected
and observed stream masks, received-record count, stage-specific source-drop
deltas, their aggregate, transport-drop delta, timeout, and protocol-error
state per device and in aggregate. Callback-ring drops remain separate in
`ProgramRealtimeRecordBatch::dropped`.

Loss baselines are the last watermark snapshot observed when a collection is
registered. Concurrently outstanding collections can therefore report
overlapping deltas; they are per-collection diagnostics, not a partition of a
global loss counter. Serial exact collections partition deltas naturally.
Watermark request/protocol counters remain control-plane device diagnostics and
are not mislabeled as descriptor-stage interval loss. A generation crossing is
carried by the dedicated protocol-error marker; a request that cannot be
published causes an incomplete timeout.

Collection registration is bounded and used only by the explicit exact API;
normal Finish performs no allocation or host collection bookkeeping. Exact
results are retained until their caller consumes or cancels them. An exception
in the Finish path cancels its registered state before propagating the exception.
Virtualized-unicast dispatch is outside this Blackhole route: attempting its
watermark path increments the request-loss counter instead of silently
claiming completion.

If the profiler is ineligible or inactive, the exact API returns
`profiler_inactive` rather than reporting a protocol violation. A malformed or
unexpected watermark returns an incomplete `protocol_error` result immediately
instead of consuming the caller's full timeout.

The caller must keep the mesh and command queue alive until the exact API
returns, matching the lifetime rule for other queue operations. Shutdown wakes
registered collection waits before destroying the profiler manager.

## Lifecycle

Host initialization zeros all queue indices, drop counters, sequences, stream
reset generations, termination handshake, and both local mailbox states. It
publishes the remote mailbox address, launches the reserved-core kernels, writes
their socket configuration, and only then publishes the nonzero profiler-core
NOC coordinate that activates dispatch_s and TRISC0. TRISC0 waits for that
enable word before touching the protocol. dispatch_s treats a zero coordinate as
disabled and does not publish descriptors.

On the supported Blackhole worker-dispatch route, dispatch_d owns every explicit
`CLEAR_STREAM`; `process_dispatch_s_wait_cmd()` does not execute. dispatch_d and
dispatch_s are co-located on the same worker tile and already receive the same
`REALTIME_PROFILER_MSG_ADDR`. After the required worker wait, dispatch_d:

1. clears the hardware stream counter through the existing stream update;
2. executes a RISC I/O-to-memory fence so the clear is ordered before shared-L1
   publication;
3. increments `stream_reset_generation[stream - first_stream_used]` in the
   shared profiler L1 block;
4. executes `fence w,w` and performs no profiler wait.

The epoch is necessarily published just after the hardware clear. The preceding
worker wait guarantees that every valid old-epoch target is already complete,
so TRISC0 cannot expose an unfinished valid descriptor in that short window.

This covers sub-device-manager loads, event/reset paths, and host 32-bit
completion-count wrap because all of them ultimately execute dispatch_d's
`process_wait(... CLEAR_STREAM ...)` path. Natural 17-bit counter rollover does
not execute that path and therefore does not change the generation.

Before publishing a descriptor, dispatch_s snapshots the shared reset
generation into it. TRISC0 maintains the local adopted generation per stream
and checks the shared generation before its counter and descriptor scan. On a
change it consumes and counts old-generation descriptors, adopts the new
generation, and only then consumes matching descriptors. Every descriptor must
match the adopted generation.

TRISC0 calls `invalidate_l1_cache()` before reading the shared generation and
producer index, and again before reading descriptor payload words. dispatch_s
performs the same two-step invalidation when it consumes TRISC0 records. Host
initialization starts both sides at generation zero.

Termination occurs only after application work is quiesced. dispatch_s sets a
dedicated termination request that the mailbox-B IDLE acknowledgement cannot
overwrite, then waits for TRISC0's stopped acknowledgement with a device-cycle
deadline. It executes a terminal forwarding loop capped by both the completed
ring capacity and a device-cycle deadline. Each iteration invokes the same
nonblocking one-item service. The bounded loop may poll an occupied mailbox B
for its IDLE acknowledgement until the cycle deadline; this wait is permitted
only after application work has quiesced and lets the last accepted record
advance. After a successful observer stop, remaining completed records and
start descriptors are counted in
dedicated terminal-loss counters. Observer-stop timeout has its own counter.
The reserved profiler BRISC observes the dedicated request, sets its ring
terminate flag, and the NCRISC drains accepted ring entries before exit.

A timeout or terminal loss is reported as an incomplete/lossy collection; it
must not trigger a D2H tensor fallback or host-duration substitution.

Sequence and unsigned producer/consumer differences use natural 32-bit
wraparound. Ring capacity is far below 2^31, so the standard unsigned-distance
full/empty tests remain unambiguous.

## Ordering proof obligations

Milestone 1 must preserve these edges:

1. dispatch_s descriptor words happen before descriptor producer-index publish;
2. TRISC0 invalidates its Blackhole L1 cache before reading the producer index
   and invalidates again before reading descriptor words;
3. TRISC0 interval words happen before completed-ring producer-index publish;
4. dispatch_s invalidates its Blackhole L1 cache before reading the
   completed-ring producer index and invalidates again before reading payload
   words;
5. mailbox-B words are visible before the inline NOC state notification;
6. BRISC completes the NOC read before acknowledging mailbox B idle;
7. BRISC ring-slot data is visible before its producer-index publish;
8. NCRISC completes D2H writes before advancing the local consumer index;

Milestone 2 must add and separately prove watermark ordering after every
accepted interval through its snapshotted producer index.

The Blackhole implementation will use explicit RISC `fence w,w` instructions at
local-L1 publication points and existing NOC read/write barriers at NOC
handoffs. Any weaker sequence requires device evidence and a new review.

## Resource and performance gates

Milestone 1 limits are:

- dispatch_s profiler-message L1: at most 8 KiB total;
- reserved profiler-core L1: no increase over 262,336 B;
- dispatch_s NCRISC text: at most +1,536 B;
- dispatch_s TRISC0 text: at most +2,048 B;
- profiler BRISC text: at most +1,536 B;
- profiler NCRISC text: no intentional increase in Milestone 1;
- no added NOC transaction when profiling is disabled;
- no loop that waits for profiler consumer progress on dispatch_s or TRISC0
  during steady-state execution; the fixed item/cycle-budget terminal handoff
  after application quiescence is the only exception;
- disabled dispatch-throughput regression at most 0.5% outside paired-run
  noise;
- enabled blank-program dispatch-throughput regression at most 2.0% outside
  paired-run noise.

The throughput thresholds are qualification gates. They do not permit a device
timing error, silent loss, or application-dispatch wait in exchange for speed.

## Rejected alternatives

### Keep the current dispatch_s drain and call it only at Finish

Rejected because ordinary Finish would remain proportional to record count and
dependent on BRISC/NCRISC/host progress. It also does not remove the exact TRISC
completion-count wait.

### Have the reserved profiler BRISC continuously poll dispatch_s L1

Rejected because it adds continuous NOC reads while the profiler is always on.
Bounded opportunistic forwarding reuses dispatch_s progress points and sends NOC
traffic only for actual records.

### Have dispatch_s capture completion after `wait_for_workers()`

Rejected because no subsequent command may arrive near the real completion
time. A flush could therefore produce an end timestamp delayed by arbitrary host
idle time. TRISC0 is the existing device observer at the completion source.

### Let TRISC0 overwrite one timestamp slot per stream

Rejected because dispatch_s can start a later program after the previous worker
completion but before TRISC0's slot is consumed. Per-stream descriptor rings and
a completed-record ring make this race explicit and bounded.

### Emit one interval for every descriptor already satisfied in a TRISC0 poll

Rejected because one sampled wall-clock value cannot prove distinct completion
times for multiple programs. Ambiguous intervals are counted as lost.

## Milestone 1 implementation surface

Expected files are limited to:

- `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`;
- `tt_metal/impl/dispatch/kernel_config/dispatch_s.cpp`;
- `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`;
- `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` for the nonblocking explicit
  `CLEAR_STREAM` generation publication;
- `tt_metal/impl/dispatch/kernels/cq_realtime_profiler_dispatch_subordinate.hpp`;
- `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`;
- `tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp`;
- `tt_metal/impl/dispatch/kernels/realtime_profiler.hpp` for removal of
  obsolete ping-pong helpers;
- `tt_metal/impl/dispatch/kernels/realtime_profiler_protocol.hpp`;
- `tt_metal/impl/dispatch/kernels/REALTIME_PROFILER.md`;
- `tt_metal/impl/dispatch/kernels/cq_commands.hpp`;
- `tt_metal/impl/dispatch/device_command.hpp`;
- `tt_metal/impl/dispatch/device_command.cpp`;
- `tt_metal/impl/program/dispatch.hpp`;
- `tt_metal/impl/program/dispatch.cpp`;
- `tt_metal/distributed/fd_mesh_command_queue.cpp` for exact worker-count
  plumbing;
- `tt_metal/distributed/realtime_profiler_manager.hpp` and
  `tt_metal/distributed/realtime_profiler_manager.cpp` for Blackhole capability
  gating, protocol-key propagation, and focused device-loss diagnostics;
- focused profiler tests and this documentation.

Host collection and public result changes belong to Milestone 2. If Milestone 1
requires edits outside this list, the reason must be documented before the diff
is reviewed.

## Milestone 1 implementation evidence

The device publication path is implemented and gated in host eligibility to
Blackhole. `CQDispatchGoSignalMcastCmd` remains 16 bytes including its command
ID: the exact launch worker count occupies one byte and `wait_stream` is narrowed
to one byte with host range checks. Worker counts above 255 are encoded as zero,
which makes the launch explicitly unprofiled instead of aborting the application.
The dispatch-core message is 5,080 bytes, an increase of 656 bytes from the
corrected 4,424-byte baseline and below the 8 KiB gate. The reserved
profiler-core layout remains 262,336 bytes.

The single 32-byte mailbox-B payload deliberately precedes the independent
termination words. This preserves its 16-byte Blackhole NOC alignment;
firmware compile-time checks validate the generated buffer-B address and
alignment. Removing the unused A payload and state avoids dead storage and
state-machine branches.

The Blackhole images produced by the focused Release test have these payloads:

| Component | Milestone 1 measured image payload | Milestone 0 gate |
| --- | ---: | ---: |
| dispatch_s NCRISC | 5,952 B text, 56 B data, 356 B BSS | 5,628 B text; +324 B |
| dispatch_s TRISC0 monitor | 1,240 B text | 900 B text; +340 B |
| reserved profiler BRISC | 912 B text | 964 B text; -52 B |
| reserved profiler NCRISC | 1,592 B text, 48 B data, 24 B BSS | no intentional source change |

The before/after image pairs use the same Blackhole worker-dispatch,
one-command-queue, eight-stream configuration. The protocol build key changes
the JIT identity so old layouts cannot reuse a current image.

Validation on the local four-chip P150b Blackhole QuietBox, firmware 19.10.0,
KMD 2.10.0, IOMMU enabled:

```text
cmake --build build_Release --target unit_tests_dispatch -j 32
  PASS

tt-metalium-validation-basic --gtest_filter='RealtimeProfilerSanity.*'
  includes disabled-close, concurrent completion, full-buffer, multi-ready,
  reset-generation, callback, finish, and trace coverage

unit_tests_dispatch \
  --gtest_filter='RealtimeProfilerSanity.ConcurrentPartitionedSubDevicesUseIndependentCompletionTargets' \
  --gtest_repeat=10 --gtest_break_on_failure
  10/10 PASS
```

The concurrent test warms the three kernel launches before measurement,
selects each independent sub-device through the dispatch stall group, and
asserts from device ticks that both short-stream intervals start and end before
the long first-stream interval ends. The third stream also exercises the go
command word that is reused as aligned NOC staging. Host timestamps are not
captured or substituted.

The unchanged 4,096-program `RealtimeProfilerStress.PeakLoadPreservesRecords`
trace completed one replay in 3.9 seconds under the same external 120-second
guard. It delivered all 16,388 expected records across four devices with zero
transport drops and a peak D2H FIFO occupancy of 66/32,768 pages. The earlier
timeout was the disabled/ineligible observer-shutdown bug: TRISC0 was launched
but its terminate request was conditional on profiler activation. Termination
is now unconditional and the disabled-close test covers this route.

Six paired 1,024-program op-to-op runs measured the final enabled path against
`TT_METAL_DISABLE_REALTIME_PROFILER=1`, reversing run order on alternating
pairs. Median totals were 52,461.5 us enabled and 52,056 us disabled (+0.78%);
means were 52,447.8 us and 52,392.5 us (+0.11%). Per-pair differences ranged
from -2,462 us to +1,428 us, so the observed delta is within paired-run noise.
This is a host
dispatch-throughput overhead signal only. Device interval durations continue
to come exclusively from device ticks. It is not a disabled-new-code versus
Milestone-0 binary comparison, so the evidence only bounds the current enabled
cost; the guarded `CLEAR_STREAM` path and activation checks are what keep the
new queue work out of an ineligible profiler run.

TRISC0 records the first sampled device tick after the completion predicate is
observed. The sampling delay is bounded by the observer loop period, but that
period has not yet been characterized on silicon. Milestone 3 must measure it
before making an absolute end-timestamp error claim; Milestone 1's concurrent
test establishes ordering from device ticks, not sub-poll-cycle accuracy.

## Milestone 2 implementation evidence

Milestone 2 implements protocol version 7. Each explicit exact collection
registers one batch ID and emits a request for every selected stream. TRISC0 snapshots the
successful interval sequence, descriptor/observer/record source loss, reset
generation, and completed-ring producer index. dispatch_s publishes the watermark only after its consumer
reaches that index, and the profiler-core BRISC adds the cumulative transport
loss snapshot. The host validates the watermark sequence against the last
received interval sequence when no transport loss explains a gap.

The public experimental collection result reports exact requested/observed
watermarks, per-device expected/observed stream masks, record count,
descriptor/observer/record/transport loss deltas, aggregate source loss,
timeout, and protocol-error state. A timeout is lossy and incomplete; it never
triggers an L1 readback, tensor transfer, or host-duration substitution.
Callback-ring loss remains callback-local.

The dispatch-core message grew by 456 B, from 5,080 B to 5,536 B, below the
8 KiB gate. The reserved profiler-core layout remains 262,336 B. Matched current
Blackhole images from the focused test are:

| Component | Milestone 1 text | Milestone 2 text | Delta |
| --- | ---: | ---: | ---: |
| dispatch_s NCRISC | 5,952 B | 7,036 B | +1,084 B |
| dispatch_s TRISC0 | 1,240 B | 1,664 B | +424 B |
| profiler BRISC | 912 B | 1,100 B | +188 B |
| profiler NCRISC | 1,592 B | 1,592 B | 0 B |

The watermark scan is out of line and guarded by a local pending-stream mask.
Before a Finish arms a request, ordinary dispatch_s profiler service adds one
mask branch rather than an inlined per-stream scan.

The full-ring test compiles an NCRISC-only saturation pause under
`TT_RT_PROFILER_RING_TEST_HOOK`. The production image contains no pause check
and remains 1,592 B text; the hook is consulted only at occupancy 4,095 or
4,096 in the test variant.

Validation on the local four-chip Blackhole P150b QuietBox, firmware 19.10.0:

- Release `unit_tests_dispatch` build passed;
- `RealtimeProfilerProtocol.*:RealtimeProfilerSanity.*`: 16/16 passed,
  including exact three-stream collection, delayed receiver timeout/recovery,
  64 back-to-back exact watermarks, empty batches, watermark-ID wrap, reset,
  stage-specific source-loss deltas, inactive status, reserved-slot interval
  loss, prompt malformed-watermark protocol error, and full-ring watermark
  retention/recovery;
- final focused stress with `TT_RT_PROFILER_SATURATION_SECONDS=0`: 16,388 of
  16,388 records, four active devices, peak FIFO 80/32,768 pages, zero
  transport drops, and no invalid intervals;
- the full two-test stress suite also passed before the final scan-size
  tightening, including slow-callback accounting. The final tightening changed
  only watermark scan placement and was revalidated by the exact collection
  and focused stress tests.

Absolute enabled/disabled overhead distributions and the completion-observer
sampling-period bound remain Milestone 3 qualification work.

The final Milestone 2 diff and the evidence above received exact `APPROVE` from
Claude Opus after review of device ordering, bounded behavior, host lifecycle,
loss semantics, tests, and production resource measurements.
