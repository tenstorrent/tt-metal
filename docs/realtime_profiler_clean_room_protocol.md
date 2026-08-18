# Blackhole Concurrent Realtime Profiler Protocol

## Status

This is the Milestone 0 decision record for
`REALTIME_PROFILER_CLEAN_ROOM_IMPLEMENTATION_PLAN.md`.

Milestone 1 evidence found that its original 0.5% cross-build host-throughput
gate is smaller than a comment-only rebuild control can resolve. The corrected
adjudication under Resource and performance gates was accepted by the product
owner on 2026-08-18.

It defines a Blackhole-only extension of the existing continuous realtime
profiler. It does not define model integration, operation scheduling, Wormhole
feature support, multi-command-queue behavior, remote-chip transport, trace
replay, or an explicit watermark/collection API.

## Support matrix

Concurrent intervals are active only when all of these are true:

- Blackhole architecture;
- one hardware command queue;
- local MMIO device with the existing D2H socket prerequisites;
- worker dispatch with dispatch_d and dispatch_s co-located;
- no virtualized-unicast completion workaround;
- at most eight dispatch streams and every stream index fits the go-command
  field validated by the host.

The manager records one of these inactive reasons before launching the observer:

```text
not_initialized
disabled_by_environment
unsupported_architecture
multiple_hardware_command_queues
non_mmio_device
iommu_unavailable
non_worker_dispatch
distributed_dispatcher
virtualized_unicast
no_reserved_profiler_core
insufficient_l1
kernels_nullified
socket_initialization_failed
```

Unsupported execution continues without concurrent records. It never creates
the observer TRISC, emits a profiler command from `Finish`, or waits for
profiler state. Wormhole realtime profiling is intentionally disabled for this
feature and its baseline monitor TRISC is removed. No Wormhole compatibility is
required. The product owner explicitly approved excluding Wormhole build,
execution, and throughput qualification while narrowing the work to the
Blackhole realtime feature. Shared host ABI assertions still apply, but no
Wormhole hardware or compile artifact is a milestone gate.

`TT_METAL_DISABLE_REALTIME_PROFILER=1` is latched when `MetalContext` is
constructed. It produces `DisabledByEnvironment`, reserves no profiler core,
and creates no observer or reserved-core kernel. It is the isolated disabled
mode used by the locked Milestone 1 throughput comparison.

Trace replay is not an initialization-time capability because capture and
replay occur after the device has opened. The manager remains active for normal
commands, but trace-captured go commands are deliberately unprofiled: both
`runtime_program_id` and `profiler_num_workers` are written as zero in the
captured command. Replay therefore publishes no descriptor, while ordinary
commands before and after replay remain profileable. This avoids applying a
capture-time `wait_count` after trace counter rebasing. Milestone 1 tests zero
trace records, correct replay execution, and resumed ordinary profiling.

## Baseline failure

The baseline correlates runtime IDs and timestamps through two global A/B
mailboxes plus a 32-entry program-ID FIFO. dispatch_d appends to that FIFO in an
unbounded loop. dispatch_s pops at command-loop rate and assigns the value to
whichever A/B slot is currently writable.

That protocol has three independent problems:

1. FIFO pressure can block dispatch_d even though profiling is observational.
2. One global A/B slot cannot represent independent sub-device completion
   order.
3. TRISC0 retains only the latest observed completion transition, so a later
   transition can overwrite an earlier one before dispatch_s consumes it.

`CQ_DISPATCH_CMD_RT_PROFILER_FLUSH` makes ordinary `Finish` wait for and pulse
the final A/B record. The clean protocol deletes that command and does not make
queue completion responsible for profiler delivery.

## Timing contract

This section is the Milestone 2 interval contract. Milestone 1 deliberately
publishes only `runtime_program_id` and the device `start_tick`, with the
public end timestamp set to zero. The inherited observer endpoint is associated
with launch N-1 while the Go-carried ID is associated with launch N; exposing
them together would create a false interval.

Every accepted interval contains two 64-bit raw Blackhole wall-clock ticks:

- `start_tick`: sampled by dispatch_s after the prior same-stream worker wait,
  immediately before descriptor publication and go-signal issuance;
- `observed_completion_tick`: sampled by TRISC0 on the first poll that observes
  the descriptor completion target satisfied.

The reported value is a device-produced correlated interval, not an exact
worker-kernel boundary duration:

```text
reported duration = worker execution + go-delivery/fanout lead + observer polling lag
```

Host enqueue, callback, synchronization, and timeout timestamps are never used
for either interval endpoint. A missing or lossy interval remains missing; it
is never replaced by host duration or tensor readback.

The completion target is:

```text
target = wait_count + profiler_num_workers
```

in the stream counter's existing modular domain. The host already knows
`num_workers` per launch. For a profiled launch, a value that cannot be encoded
is marked unsupported, counted, and launched without a profiler descriptor.

Blackhole worker completion counters are 17-bit. Targets are masked to that
domain and compared with the existing shifted half-range comparison. The
half-range contains 65,536 values, so the largest unambiguously positive
forward distance is 65,535 completion contributions. A launch contribution is
at most 255, and descriptor capacity is four, so a valid target cannot be more
than 1,020 contributions ahead of the sampled base. Host and firmware validate
that bound; a larger delta is `unsupported_launch`, never a retained head that
could become falsely satisfied after natural rollover.

## Command ABI

`CQDispatchGoSignalMcastCmd` remains one 16-byte `CQDispatchCmd` including its
command byte. It carries:

- existing go signal and multicast/unicast routing;
- completion `wait_count`;
- an 8-bit stream index;
- an 8-bit profiled worker contribution;
- a 16-bit runtime program ID.

Compile-time size/offset assertions and host range validation are mandatory.
Runtime ID zero means intentionally unprofiled traffic and is checked before
worker-contribution validation; it does not increment `unsupported_launch`.
For a nonzero runtime ID, a zero or unencodable worker contribution is counted
as unsupported and launched without a descriptor. Values outside the existing
16-bit runtime-ID contract are rejected or explicitly truncated under the
pre-existing issue; they may not alias silently inside the new protocol.

The program ID is carried only by this go command. The old FIFO, indices,
append/pop helpers, initialization, wait loop, and test exclusion are deleted.

dispatch_s copies `runtime_program_id`, `profiler_num_workers`, `wait_stream`,
and every other profiler field to locals before the command buffer is reused as
aligned go-signal storage. The existing writes to `cmd_ptr[storage_offset]` and
`cmd_ptr[0]` may overwrite any byte of the 16-byte command; no profiler field is
read lazily after that point.

Narrowing `wait_stream` also changes the shared Wormhole command layout even
though Wormhole profiling is disabled. Host range assertions apply on every
architecture. Per explicit product-owner scope, Wormhole build and throughput
qualification for this shared ABI change are not milestone gates.

## State layout and ownership

All mutable queues are single-producer/single-consumer.

| State | Sole producer | Consumer |
| --- | --- | --- |
| Per-stream start descriptor ring | dispatch_s NCRISC | dispatch_s TRISC0 |
| Register-space descriptor publication signal | dispatch_s NCRISC | dispatch_s TRISC0 |
| Completed interval ring | dispatch_s TRISC0 | dispatch_s NCRISC |
| Register-space completed publication signal | dispatch_s TRISC0 | dispatch_s NCRISC |
| Transport mailbox | dispatch_s NCRISC | reserved profiler BRISC |
| Reserved-core ring | reserved profiler BRISC | reserved profiler NCRISC |
| D2H pages | reserved profiler NCRISC | host receiver |

There is no cross-RISC atomic read/modify/write state and no lock.

Runtime control and counter ownership is equally strict:

| State | Sole runtime writer | Readers |
| --- | --- | --- |
| Per-stream generation, reset epoch, `dispatch_d_ready` | dispatch_d | dispatch_s, TRISC0, host diagnostics |
| Descriptor producer indices, descriptor epoch, `descriptor_full`, `unsupported_launch`, `terminal_descriptor`, `dispatch_s_ready` | dispatch_s | TRISC0 or host diagnostics |
| Descriptor consumer indices, completed producer index/epoch, `reset_descriptor`, `observer_coalesced`, `stuck_head`, `completed_record`, `terminal_record`, `observer_ready` | TRISC0 | dispatch_s or host diagnostics |
| Completed consumer index | dispatch_s | TRISC0, host diagnostics |
| `device_ring` | reserved profiler BRISC | reserved profiler NCRISC, host diagnostics |
| Callback-ring dropped count | host callback fanout | callback batch consumer |
| Activation/capability state | host manager | dispatch_s, TRISC0, public API |
| Observer-stop timeout diagnostic | host manager | public API |

### Cold initialization

The host zeros the whole profiler L1 region before any dispatch kernel is
launched, exactly as `DispatchSKernel::ConfigureCore` does for the baseline
mailbox. That pre-launch construction write cannot race a device RISC. Runtime
ownership then begins and no RISC clears another RISC's fields:

- dispatch_d initializes stream generations, its loss fields, stream-8 scratch
  3, and finally publishes `dispatch_d_ready`;
- dispatch_s initializes every descriptor producer index, the completed-ring
  consumer index, its loss fields, stream-8 scratch 5, and finally publishes
  `dispatch_s_ready`;
- TRISC0 initializes every descriptor consumer index, the completed-ring
  producer index, its loss fields, stream-8 scratch 4, and finally publishes
  `observer_ready`;
- the reserved profiler core initializes its existing transport/ring fields and
  its `device_ring` loss counter before reporting socket readiness.

Payload words need no initialization because no consumer can read them until
its producer publishes a nonzero index. Each ready flag is written by exactly
one RISC after its payload/index and register initialization is visible. During
the existing host initialization handshake, the manager waits with its finite
initialization timeout for all three dispatch-core ready flags and the
reserved-core socket readiness. It then writes the existing activation field
once. Mesh open returns with the capability active only after this handshake;
otherwise the observer is terminated and the device capability records a
specific inactive reason. dispatch_s and TRISC0 park on activation before
touching queue state, so stale persistent scratch values cannot activate the
protocol. Reload repeats the same ordered construction handshake.

The host is the sole pre-launch initializer; the named device RISC is the sole
runtime writer of each field. This explicitly replaces the baseline's competing
dispatch_d/TRISC mailbox clears.

For the M1 legacy mailbox, the host writes the remote state address before the
nonzero profiler-core NOC coordinate, which is the activation release. If D2H
socket construction fails after dispatch_s was compiled with profiler support,
the host writes a permanent disabled sentinel. dispatch_s resolves that
sentinel once and does not poll activation on later commands. M1 removes the
legacy completion observer entirely; its TRISC0 image is a stub, and only
independent TRISC1 telemetry remains.

### Start descriptors

Each of eight streams owns a power-of-two ring with compile-time capacity four.
Each five-word descriptor contains:

```text
runtime_program_id
start_tick_hi
start_tick_lo
completion_target
generation
```

Capacity four is the default because saving approximately 320 bytes by using
depth two leaves no observer slack after one completed-but-unobserved launch.
Milestone 2 measures both depths at the shortest supported launch and may reduce
the default only if the locked loss and L1 gates pass.

dispatch_s first copies all command fields to locals. For a multicast launch it
programs the existing NOC state, waits for prior same-stream workers, then does
the local-only descriptor publication immediately before
`cq_noc_async_write_with_state`. Local L1 stores do not touch the protected NOC
command buffer. For a unicast-only launch it waits first, publishes immediately
before the unicast loop, and uses one start tick for the fanout. A launch with
both forms is timestamped before the first multicast go.

Publication writes payload, executes `fence w,w`, publishes the L1 producer
index, and increments the stream-8 descriptor epoch last. If the ring is full it
increments `descriptor_full` and issues the go signal without a descriptor.
There is no retry. A fatal NOC failure after publication is a device execution
failure, not a recoverable profiler event; ordinary successful go issuance
cannot leave the descriptor behind without launching the program.

### Register-space publication signal

Blackhole has no uncached L1 alias. Empty-path readiness therefore uses
NOC-overlay register space, which TRISC0 already reads without D-cache
invalidation.

The selected signals are:

```text
stream 8, STREAM_SCRATCH_4_REG_INDEX: completed-publication epoch
stream 8, STREAM_SCRATCH_5_REG_INDEX: descriptor-publication epoch
```

Blackhole scratch registers 0–5 exist on streams 0–3 and 8–11 and store 24 data
bits. The dispatch completion streams are 48–55. Repository search finds no
dispatch-kernel use of stream 8 scratch 3/4/5. Milestone 0's on-device ownership
test passes values from NCRISC to TRISC0 through all three registers, passes an
acknowledgement from TRISC0 back to NCRISC through scratch 4, and restores the
original values.
These registers are reserved by the profiler protocol build and never used on
Wormhole.

dispatch_s is the sole writer of the descriptor epoch and TRISC0 is the sole
writer of the completed epoch. Both increment modulo 2^24 after publishing the
corresponding L1 producer index. Half-range modular comparison has an
unambiguous range of 8,388,608 publications. Epochs are wakeup hints only: the
authoritative queue producer/consumer indices remain 32-bit values in L1.

TRISC0 owns its active mask and consumer indices locally. While idle it reads
only the descriptor epoch. When it changes, TRISC0 invalidates L1, reads all
eight per-stream producer indices, and derives its active mask. It then polls
only active streams. dispatch_s applies the symmetric completed-epoch rule.

### Completion observation

For each active stream, TRISC0 compares the head target to the authoritative
`STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX` count using the existing
wrap-aware comparison.

When one target is satisfied, TRISC0 captures one end tick. If several queued
targets are already satisfied by that sample, only the newest receives the
tick. Older descriptors are consumed and counted in `observer_coalesced`; one
sample may not be presented as several exact completion times.

An immediately previous-generation descriptor reached through the ordered reset
path follows the reset-observation drain rule below. A descriptor older than
that, or one without the ordered epoch transition, is consumed and counted in
`reset_descriptor`. A same-generation unsatisfied head in a full ring increments
`stuck_head` once per episode but is not discarded by a wall-clock timeout; it
may be a valid long-running program. Later descriptor loss remains explicit and
application dispatch remains unblocked.

### Completed interval ring

TRISC0 publishes successful intervals into a compile-time 64-entry SPSC ring.
Producer and consumer are monotonic `uint32_t` indices; the ring is full when
their unsigned distance is 64, so all 64 array slots are usable. Each interval
contains exactly eight words:

```text
word 0: start_tick_hi
word 1: start_tick_lo
word 2: runtime_program_id[15:0] | generation_low16[31:16]
word 3: schema_version[7:0] | record_type[11:8] | cq_id[15:12] |
        stream_id[23:16] | reserved[31:24]
word 4: end_tick_hi
word 5: end_tick_lo
word 6: successful_publish_sequence
word 7: cumulative_source_loss_low32
```

The authoritative generation remains 32-bit in the descriptor/control state.
The wire exposes its low 16 bits for correlation; a callback consumer must use
the tuple `(chip_id, cq_id, stream_id, generation_low16, sequence)` and treat a
generation discontinuity as a boundary. More than 65,535 resets between
successive delivered records is outside the supported observation window and
is surfaced by the final detailed diagnostic snapshot. Host structs widen the
decoded fields to `uint32_t`/`uint64_t` for source compatibility and monotonic
extension.

The 64-entry capacity is 2,048 bytes and is fixed by compile-time assertions:
it is a power of two and is at least twice
`num_streams * descriptor_capacity = 8 * 4 = 32`. Therefore it can hold every
simultaneously accepted descriptor completion plus one full outstanding set of
transport backlog. dispatch_s services exactly one record per safe service
point; there is no variable or draining loop. The command-loop, page-wait,
sync-wait, and post-command service points make forward progress under normal
dispatch, while capacity pressure still drops and counts rather than blocks.
Milestone 2's shortest-program/all-eight-stream test must show zero
`completed_record` loss under normal transport; injected downstream stalls are
expected to fill the ring and prove exact counted loss.

The sequence advances only when the completed ring accepts the record. A full
ring increments `completed_record` and consumes the descriptor. Word 7 at this
stage contains TRISC0-owned cumulative observer loss. Payload is published
before producer index with `fence w,w`.

TRISC0 also publishes a monotonic completed-record signal in register space.
dispatch_s checks that signal first. A clear signal returns with no L1
invalidation. A pending signal permits one invalidation and one bounded service
step for exactly one record.

### NOC command-buffer safety

No profiler NOC operation may execute between
`cq_noc_async_write_init_state` and `cq_noc_async_write_with_state` in the
multicast go path. Blackhole's L1-destination inline-write workaround can use
`NCRISC_WR_CMD_BUF` regardless of the requested dispatch_s buffer; inserting a
profiler notification in this window can silently corrupt the go signal.

Profiler service points are enumerated in code and tested. Saving/restoring
command-buffer registers is defense in depth, not permission to service inside
the protected stateful-write pair.

The exact steady-state service points are:

1. once at the top of the dispatch_s command loop, before acquiring the next
   command page;
2. once per iteration while waiting for command-buffer pages;
3. once per iteration in the dispatch_d sync-semaphore wait, before multicast
   NOC state is initialized;
4. once after a command has completed all stateful go writes and before its page
   is released.

No NOC-forwarding service occurs inside `wait_for_workers`, because the default
multicast path calls that wait between init-state and with-state. TRISC0 remains
free to publish into the bounded completed ring during that wait. The next safe
service point forwards it without affecting the measured end tick.

## Transport behavior

The existing reserved profiler BRISC/NCRISC, 4096-entry 64-byte ring, D2H socket,
receiver thread, callback registration, and kernel-source lookup are retained.

dispatch_s forwards at most one ready interval per normal service action:

1. return if transport mailbox is occupied;
2. return if the register-space completed signal is clear;
3. invalidate and copy one complete record;
4. add dispatch_s-owned launch/descriptor loss to word 7 in the outbound copy;
5. publish mailbox payload before its state notification;
6. advance the completed-ring consumer and return.

The reserved BRISC never waits for interval ring space. A full ring increments
`device_ring` and acknowledges the mailbox so dispatch can continue. For every
record that the ring does accept, BRISC adds its cumulative `device_ring` count
to word 7 of the outbound ring copy before publishing the write index. Thus a
sentinel accepted after pressure carries all earlier upstream and ring loss;
TRISC0 is not incorrectly treated as the owner of downstream loss. All additions
are modulo 2^32. The host extends successive low-32 snapshots to 64 bits using
unsigned deltas; a producer may not discard 2^32 intervals without delivering a
record or terminal diagnostic snapshot. NCRISC and the D2H socket may drain
independently; host pressure becomes counted loss and cannot propagate into
dispatch.

Ordinary `Finish` emits no realtime-profiler command. The asynchronous callback
and normal teardown drain deliver final accepted records. Teardown has explicit
item and cycle budgets and reports terminal descriptor, terminal record, and
observer-stop failures. Terminal counters that arise after the last accepted
record remain available through the explicit capability/diagnostic query; close
does not add an automatic L1 read or a fallback D2H path.

## Reset protocol

Completion-counter reset uses the existing command-order and next-go sync
semaphore as its carrier. In the supported co-located topology dispatch_d owns
`CLEAR_STREAM`; dispatch_s's separate-core `process_dispatch_s_wait_cmd` never
executes.

For one stream, dispatch_d:

1. waits for the requested old completion target;
2. clears the 17-bit authoritative completion register;
3. increments the shared-L1 stream generation and executes the required
   I/O-to-memory and write fences;
4. increments a dedicated 24-bit reset epoch in stream 8 scratch register 3;
5. only afterward processes the next ordered
   `CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL`, which releases dispatch_s's
   per-stream sync semaphore for the next go.

dispatch_s cannot publish the next descriptor until that semaphore advances.
After it advances, dispatch_s compares the register-space reset epoch, invalidates
L1 if it changed, and snapshots the new generation into the descriptor. TRISC0
also checks the reset epoch before scanning active descriptors. On an orderly
one-generation transition it knows the clear command waited through every old
target. It captures one reset-observation tick, coalesces multiple satisfied old
descriptors exactly like a late normal poll, and attempts to emit the newest as
`record_type=reset_observed`; that record is explicitly an upper-bound device
observation whose polling lag includes reset handling. Only descriptors older
than the immediately previous generation, or descriptors encountered without
the ordered reset epoch, are discarded as `reset_descriptor`. TRISC0 then adopts
the new generation.

TRISC0 can observe the cleared completion register before the reset epoch, but
no new descriptor can enter that window. The clear command has already waited
for every valid old target, and the observer never treats an apparent backward
17-bit count as forward progress for that head. This reset-observation rule
preserves the last accepted interval across normal close without making reset
wait for the profiler. An adversarial Milestone 2 hook
pauses dispatch_d between clear and generation publication and proves the next
go remains behind the sync semaphore and no interval is fabricated.

Any topology without this ordered command/sync carrier is ineligible.

Natural completion-counter rollover is not a reset and uses the existing
half-range comparison. Generation and queue indices use natural unsigned
wraparound with capacities below half range.

## Loss model

Device counters are monotonic and distinguish:

```text
descriptor_full
unsupported_launch
reset_descriptor
observer_coalesced
stuck_head
completed_record
terminal_descriptor
terminal_record
observer_stop_timeout
device_ring
```

`stuck_head` and `observer_stop_timeout` are diagnostics, not automatically
added to the lost-interval total. The aggregate source-loss value is defined as
the sum of counters that each represent discarded intervals. A record carries
the cumulative aggregate assembled in stages: TRISC0 stamps observer loss,
dispatch_s adds launch/descriptor loss to its mailbox copy, and the reserved
BRISC adds device-ring loss to its ring copy. No RISC writes another RISC's
counter, and a downstream loss is never expected to appear in an earlier-stage
record.

Detailed counters remain in their owning device L1 blocks and are exposed
through the explicit per-device diagnostic query. The callback batch adds
per-device snapshots containing the newest aggregate record value for that
chip. This keeps the interval wire record at eight words while separating
device-source loss from the existing `ProgramRealtimeRecordBatch::dropped`,
which continues to mean host callback-ring loss only. A caller that needs a
terminal loss audit invokes the diagnostic query after device completion; the
query never fabricates or replaces an interval.

An all-records-dropped stress test releases pressure and sends a final sentinel;
the sentinel's aggregate snapshot proves the earlier loss. No explicit batch
watermark is required.

## Public API

Existing APIs remain source-compatible:

```text
RegisterProgramRealtimeProfilerCallback
UnregisterProgramRealtimeProfilerCallback
IsProgramRealtimeProfilerActive
```

The final additions are fixed here for Milestones 1–3:

```cpp
enum class ProgramRealtimeProfilerInactiveReason : uint8_t {
    None,
    NotInitialized,
    DisabledByEnvironment,
    UnsupportedArchitecture,
    MultipleHardwareCommandQueues,
    NonMmioDevice,
    IommuUnavailable,
    NonWorkerDispatch,
    DistributedDispatcher,
    VirtualizedUnicast,
    NoReservedProfilerCore,
    InsufficientL1,
    KernelsNullified,
    SocketInitializationFailed,
};

struct ProgramRealtimeProfilerLossCounts {
    uint64_t descriptor_full;
    uint64_t unsupported_launch;
    uint64_t reset_descriptor;
    uint64_t observer_coalesced;
    uint64_t stuck_head;
    uint64_t completed_record;
    uint64_t terminal_descriptor;
    uint64_t terminal_record;
    uint64_t observer_stop_timeout;
    uint64_t device_ring;
};

struct ProgramRealtimeProfilerDeviceLossSnapshot {
    uint32_t chip_id;
    uint64_t cumulative_source_dropped;
};

struct ProgramRealtimeProfilerDeviceCapability {
    uint32_t chip_id;
    bool active;
    ProgramRealtimeProfilerInactiveReason inactive_reason;
    ProgramRealtimeProfilerLossCounts loss;
};
```

`ProgramRealtimeRecord` retains every baseline field and appends:

```cpp
uint32_t command_queue_id;
uint32_t dispatch_stream;
uint32_t generation;
uint32_t sequence;
uint32_t schema_version;
uint32_t record_type;
uint64_t cumulative_source_dropped;
```

`generation` is the zero-extended low 16 bits carried in wire word 2.
`cumulative_source_dropped` is the host-extended monotonic value derived from
wire word 7. The manager resets both extension bases at device initialization
and treats a terminal detailed snapshot as authoritative.

`ProgramRealtimeRecordBatch::dropped` retains its host-callback-ring-only
meaning and appends:

```cpp
std::span<const ProgramRealtimeProfilerDeviceLossSnapshot> device_loss;
```

The manager derives that span from the newest cumulative source-loss value per
chip in the delivered batch. Detailed stage counters are deliberately not
duplicated in every interval. The single new query is:

```cpp
std::vector<ProgramRealtimeProfilerDeviceCapability>
GetProgramRealtimeProfilerDeviceCapabilities();
```

It refreshes detailed counters with an explicit diagnostic L1 read. That is
metadata inspection, not an operation-duration fallback, and is never invoked
in the dispatch or callback hot path. `IsProgramRealtimeProfilerActive()` is
implemented as the compatibility aggregate over these per-device states.

There is no `FinishAndCollectProgramRealtimeProfiler`, watermark record, control
slot, or outstanding collection registry.

## Ordering obligations

Implementation review must prove every edge:

1. pre-launch host zero before any device RISC starts;
2. each RISC's owned initialization before its ready flag;
3. every required ready flag before host activation and mesh-open success;
4. descriptor payload before descriptor producer index;
5. descriptor producer index before register-space publication signal;
6. TRISC0 sees publication signal before invalidating/reading descriptor L1;
7. completed payload before completed producer index;
8. completed producer index before register-space completed signal;
9. dispatch_s sees completed signal before invalidating/reading record L1;
10. staged loss composition before mailbox/ring publication;
11. mailbox payload before NOC state notification;
12. BRISC NOC read completion before mailbox acknowledgement;
13. reserved-ring payload before write-index publication;
14. NCRISC D2H write completion before reserved-ring read-index advance;
15. reset counter clear before new generation acceptance;
16. no profiler NOC operation inside the protected go-signal stateful-write
    window.

## Resource and performance gates

The exact clean baseline and commands are recorded in
`docs/realtime_profiler_clean_room_milestone0_evidence.md`.

Locked gates:

- dispatch-core profiler message: at most 8 KiB;
- reserved profiler-core L1 layout: no increase over 262,336 bytes;
- dispatch_s NCRISC text: no more than +1,536 bytes over matched baseline;
- Blackhole observer TRISC0 text: no more than 2,048 bytes total;
- reserved profiler BRISC text: no more than +1,024 bytes;
- reserved profiler NCRISC: no intentional source or image-size increase;
- unsupported/disabled launch: zero new invalidation and zero NOC transaction;
- enabled-idle observer: one register-space publication check, no all-stream
  scan and no L1 invalidation;
- accepted launch: bounded O(1) descriptor publication;
- active observation: bounded O(active streams);
- no profiler-owned steady-state wait or retry loop;
- matched disabled throughput regression: paired bootstrap upper 95% confidence
  bound no more than 0.5%;
- matched enabled throughput regression: paired bootstrap upper 95% confidence
  bound no more than 2.0%;
- zero unexplained or silent loss;
- depth four must have zero descriptor loss in the shortest-program two-stream
  qualification; depth two may replace it only if it also has zero loss across
  repeated stress and materially improves a locked resource metric;
- Milestone 2 observer creation is source-gated by the eligible-Blackhole
  decision; Milestone 1 has no completion observer;
  Wormhole build and runtime evidence are intentionally not gates.

Device-cycle counters, invalidation counts, observer scan latency, L1/image
deltas, and ABBA throughput samples are required evidence. Host throughput is
corroborating evidence only.

### Accepted Milestone 1 adjudication correction

The final Milestone 1 comment-only baseline rebuild produced a paired bootstrap
upper 95% bound of 0.6083%, so the 0.5% cross-build host gate above is not
resolvable by the measurement machinery. The same-binary enabled comparison
had a 1.0168% median but a 2.5584% upper bound due process outliers, so the 2%
host ceiling is likewise unresolved. Neither is silently waived or called a
pass. Per product-owner approval on 2026-08-18, Milestone 1 is adjudicated by:

- disabled GO-tail device-cycle median no slower than the exact clean baseline;
- enabled full GO-tail device-cycle increment measured and reported;
- same-binary and cross-build host timing retained as descriptive evidence,
  with every sample and the no-op rebuild control shown beside it.

Later milestones retain device cycles as primary evidence and must add their
own mechanism-specific gates before implementation.

## Rejected alternatives

### Retain the program-ID FIFO

Rejected because its producer has an unbounded full-loop and the go command can
carry identity directly.

### Retain A/B slots per stream

Rejected because a completed interval can be overwritten before downstream
consumption and capacity/loss cannot be represented cleanly.

### Depth-two descriptors by default

Rejected pending measurement because it holds one active and one
completed-but-unobserved launch with no scheduling slack.

### Shared L1 active bitmask

Rejected because it introduces cross-RISC read/modify/write ownership and still
requires cache invalidation in the idle path. Register-space monotonic signals
match existing Blackhole stream-register practice.

### Poll every stream continuously

Rejected because the observer is always resident and unnecessary L1/register
traffic can perturb dispatch. Work scales with active streams.

### Capture completion in dispatch_s after its wait

Rejected because a subsequent command may arrive arbitrarily after actual
completion. TRISC0 is the resident observer at the authoritative counter.

### Emit all descriptors satisfied by one poll

Rejected because one sampled tick cannot prove multiple completion boundaries.

### Deterministic watermark collection

Rejected because no production consumer requires it. Expected runtime IDs,
successful sequences, continuous loss snapshots, and a finite test-only host
wait provide deterministic tests without adding device control state or making
`Finish` part of profiling.
