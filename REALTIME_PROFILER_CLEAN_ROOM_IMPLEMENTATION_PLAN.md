# Blackhole Concurrent Realtime Profiler: Clean-Room Implementation Plan

**Scope:** realtime profiler only
**Target:** Blackhole, single hardware command queue, supported local MMIO dispatch topology
**Reference implementation:** `335393399b602a7d090c5cd4b3ac7652acbfda0f`
**Clean baseline:** `5fa57f6d18bdc46e0d623a6f86a1b7d4bfcc547b` (`20e839e1ff9032573690c031751ad58b5e967fed^`), the exact pre-concurrent-profiler state
**Status:** Milestones 0 and 1 approved by Claude Opus; Milestone 1 device-cycle adjudication approved by the product owner on 2026-08-18; the combined Milestone 2/3 implementation and evidence are complete and locally qualified

## 1. Purpose

Reimplement concurrent sub-device profiling as a small extension of the
lightweight, always-on realtime profiler rather than as a second profiling
system.

The implementation must produce correctly correlated, device-generated timing
intervals for programs that overlap on independent Blackhole sub-devices. It
must never delay application dispatch to preserve a profiler record, wait for
host collection, or silently replace a missing device measurement with a host
duration.

The existing implementation at `335393399b6` is working evidence, not the code
template. It will be used to retain validated tests, protocol lessons, and
measurements while removing its redundant but live blocking FIFO, broad
polling, oversized collection state, and unsupported-configuration gaps.

The clean baseline is not a profiler-free tree. It already contains the D2H
socket, reserved profiler tensix, bounded ring, host receiver, callback path,
and the original program-ID FIFO. The transport and callback subsystems are
inherited where their invariants still hold. The FIFO and the global A/B
start/end correlation protocol are explicitly replaced. None of the concurrent
runtime changes from `20e839e1` are inherited by assumption.

## 2. Goals

### 2.1 Correct device timing

- Correlate each accepted interval with device, command queue, invocation,
  sub-device stream, and schema version.
- Capture both timestamps from the Blackhole device clock.
- Correctly handle cross-stream out-of-order completion.
- State the measurement error honestly:
  - start is captured immediately before go-signal issuance;
  - end is the first observer timestamp after the stream completion target is
    observed;
  - reported duration therefore includes go-delivery lead and observer polling
    lag and is not an exact worker-kernel boundary measurement.
- Measure and publish bounds or distributions for both sources of timing skew.

### 2.2 Realtime behavior

- No profiler-owned loop or wait in the application dispatch path.
- Every profiler operation on a launch is constant-time and bounded.
- Capacity exhaustion drops profiler data, increments a visible counter, and
  never affects program execution.
- Disabled or unsupported profiling is latched once and adds no cache
  invalidation, NOC traffic, or observer scan to each program launch.
- Enabled but idle profiling does not scan every stream continuously.
- Enabled observer work scales with active streams, not maximum streams.

### 2.3 Bounded and observable loss

- All firmware buffers and host queues have fixed capacity.
- Every distinct loss or invalid-state cause listed in Section 5.3 has a named
  monotonic counter; unrelated causes are not folded into one aggregate.
- Ordinary callback users receive a loss snapshot with records or through an
  adjacent public diagnostic API; loss is not visible only to test-only batch
  collection.
- Reset and counter wrap cannot turn stale state into a valid interval.

### 2.4 Minimal integration

- Preserve the existing profiler transport and callback pipeline where it
  already meets the contract.
- Do not add model, Sparse MLA, TopK, all-gather, core-allocation, or scheduler
  policy to this work.
- Do not implement concurrent profiling for Wormhole, Galaxy relay, multi-CQ,
  trace replay, or unsupported dispatch topologies. Trace-captured go commands
  are explicitly marked unprofiled while ordinary commands remain active.
- This feature intentionally disables the realtime profiler on Wormhole and
  removes the baseline Wormhole monitor TRISC. The product owner explicitly
  approved excluding Wormhole compatibility, execution, build, and throughput
  qualification from this Blackhole-only feature. Shared ABI assertions remain
  architecture-independent, but Wormhole is not a milestone gate.
- Reject unsupported configurations explicitly at capability/initialization
  boundaries rather than partially activating the profiler.
- `TT_METAL_DISABLE_REALTIME_PROFILER=1` is a construction-time diagnostic and
  performance gate. It reserves no profiler core, creates no reserved-core
  profiler kernel, compiles TRISC0 observer logic out of the shared compute
  kernel, and reports `DisabledByEnvironment`. Independent TRISC1 telemetry
  remains available.

## 3. Non-goals

- Exact worker-kernel start/end boundary instrumentation.
- Lossless profiling under unbounded producer or host stalls.
- Transparent support for multiple hardware command queues.
- A device watermark or explicit batch-collection API.
- A general-purpose firmware event protocol.
- A new tensor readback path or any device-to-host tensor fallback.
- Host wall-clock substitution for a missing or invalid device interval.
- Preserving the internal ABI or state layout of `20e839e1` or `335393399b6`.
- Sparse MLA integration or performance claims.

## 4. Support contract

The capability is active only when every condition below is true:

- architecture is Blackhole;
- execution uses exactly one hardware command queue;
- the command queue is local/MMIO for this phase;
- the dispatch topology has the authoritative worker-completion counters and
  reset path validated in Milestone 0;
- the number of dispatch streams fits the compile-time stream mask;
- trace-captured go commands carry zero profiler runtime ID and zero profiler
  worker contribution, so replay is deliberately unprofiled without changing
  the device's latched capability for ordinary commands.

Initialization returns a precise unsupported reason when any condition fails.
No unsupported path may report the profiler as active, create the observer
kernel, or add profiler work to dispatch/`Finish`.

The realtime profiler becomes inactive on Wormhole for this feature. No
observer kernel is intended to be created or launched there. Per explicit
product-owner direction, Wormhole build, execution, and throughput behavior are
not qualified by this plan. The new protocol is activated only for eligible
Blackhole configurations.

Trace replay is not an inactive capability reason because capture can begin
after initialization. Milestone 1 suppresses descriptors in captured commands
and proves normal profiling resumes after replay.

## 5. Timing and loss contract

### 5.1 Successful interval

A successful record contains:

```text
schema_version
device_id
cq_id
stream_id
runtime_program_id
generation_low16
start_tick
observed_completion_tick
publish_sequence
cumulative_source_dropped_low32
```

`start_tick` and `observed_completion_tick` are 64-bit raw device ticks from the
same documented clock domain. Both halves are transported explicitly and the
wrap rule is tested. Host time may be emitted separately for diagnostics, but
it can never populate either field or the reported duration.

The authoritative generation is 32-bit in shared device state; the eight-word
wire record packs its low 16 bits beside the 16-bit runtime ID. The staged
source-loss aggregate is carried modulo 2^32 and extended monotonically by the
host. The protocol decision record fixes the complete word/bit layout.

### 5.2 Measurement interpretation

For an accepted invocation:

```text
reported_duration = observed_completion_tick - start_tick
reported_duration = worker_execution + go_delivery/fanout lead + observer_poll_lag
```

The API and documentation use “device-produced correlated interval,” not
“exact kernel duration.” Qualification must measure the observer scan latency
and estimate or bound go-signal delivery lead on the supported topology.
The start tick is captured after waiting for prior work on the same stream; the
prior-worker wait is never included in the interval.

### 5.3 Loss interpretation

Every accepted descriptor produces at most one interval. The protocol retains
distinct monotonic counters for:

- descriptor-queue full;
- unsupported launch metadata for a nonzero profiler runtime ID, including zero
  or unencodable worker count; runtime ID zero is intentionally unprofiled and
  is not loss;
- stale/reset-generation descriptor;
- multiple already-satisfied descriptors coalesced into one observation;
- an unsatisfiable/stuck descriptor head;
- completed-record queue full;
- descriptors and records abandoned during bounded termination;
- observer-stop timeout/protocol failure;
- reserved device-ring full;
- host callback-ring full.

One failed interval is assigned one primary device loss cause; downstream loss
of an already accepted record is counted at the stage where it occurs.
Observer-stop timeout is a protocol diagnostic and may accompany terminal loss,
but is not added again when computing the number of lost intervals. Consumers
receive both the structured counters and an explicitly defined aggregate and
compare monotonic snapshots to determine whether a measurement range was
lossless.

The aggregate is composed at each downstream copy boundary: TRISC0 contributes
observer loss, dispatch_s contributes launch/descriptor loss, and the reserved
BRISC contributes device-ring loss. Terminal losses after the last accepted
record remain visible through the explicit per-device capability/diagnostic
query. No metadata read supplies a timing endpoint, and ordinary close adds no
fallback D2H path.

No code may retry indefinitely, wait for capacity, or overwrite an unconsumed
entry without recording loss.

## 6. Minimal device architecture

### 6.1 Launch metadata: one path, no program-ID FIFO

The existing go command carries:

- `runtime_program_id`;
- `profiler_num_workers` or the equivalent completion contribution;
- the existing stream identity.

The old program-ID FIFO is deleted in its entirety: storage, producer append,
consumer pop, initialization, and documentation. There is one source of
invocation identity, and no profiler metadata publication can block dispatch.

Any field-width trade made to retain the go-command ABI must be proved valid
with compile-time checks and supported-value validation.

### 6.2 Cold initialization and ownership

The host zeroes the complete profiler L1 region before launching dispatch
kernels. dispatch_d, dispatch_s, and TRISC0 then initialize only their owned
indices, counters, and stream-8 scratch registers and publish separate ready
flags. The existing finite host initialization handshake activates profiling
only after all dispatch-core ready flags and reserved-core socket readiness are
visible. Until activation, dispatch_s and TRISC0 park without reading persistent
queue or scratch state. Reload repeats the same handshake.

The protocol decision record names the pre-launch initializer and sole runtime
writer for every field. The baseline's competing mailbox clears are removed.

### 6.3 Per-stream descriptor queue

Each supported stream owns a compile-time-sized SPSC descriptor queue in shared
L1. The initial and default depth is four:

```text
runtime_program_id
generation
completion_target
start_tick_hi
start_tick_lo
```

The dispatch subordinate is the sole producer and the completion observer is
the sole consumer. A depth of four costs about 320 bytes more than depth two for
eight streams and provides observer-latency slack for short programs. Milestone
2 measures descriptor loss at depths two and four using the shortest supported
program with at least two streams active. The depth may be reduced only if the
locked loss and L1 budgets prove it safe.

Publication rules:

1. Read the device start tick immediately before issuing the go signal.
2. Compute the absolute stream completion target with checked arithmetic.
3. If the descriptor queue has capacity, write payload, execute the required
   Blackhole visibility primitive, then publish the producer index.
4. If it is full, increment `descriptor_drops` and continue execution without
   publishing a descriptor.
5. Issue the go signal regardless of profiler acceptance.

Milestone 0 must confirm whether descriptor publication occurs immediately
before or immediately after go issuance to minimize skew without creating a
record for a go command that cannot execute. The selected order is fixed in the
protocol document and tested.

### 6.4 Active-stream observer

Do not introduce a cross-RISC L1 read/modify/write bitmask. dispatch_s is the
single writer of a monotonic descriptor-publication signal in NOC-overlay
register space, which is not D-cache-cached on Blackhole. The observer owns its
active-stream mask locally and tracks descriptor consumer state per stream.

The selected encoding uses two otherwise-unused 24-bit scratch registers on
Blackhole stream 8: one descriptor-publication epoch written only by dispatch_s
and one completed-publication epoch written only by TRISC0. Stream 8 is outside
the dispatch completion range 48–55, and scratch registers exist on streams
8–11. When the descriptor epoch changes, the observer invalidates L1, reads all
per-stream producer indices, and derives its local active mask. The 24-bit
epochs use half-range modular comparison and are only wakeup hints; queue
indices in L1 remain authoritative.

The observer:

- parks on an activation flag when profiling is disabled;
- reads only the register-space global publication signal when profiling is
  enabled but idle;
- inspects only active streams;
- compares each head descriptor target using wrap-safe generation semantics;
- timestamps the first observation of `completion_count >= target`;
- attempts one bounded push to the completed-record queue;
- records a completed-queue drop if full;
- consumes descriptors and clears its local active bit when its consumer index
  catches the published producer index.

If one scan finds multiple descriptors on a stream already satisfied, one
sampled tick cannot represent each completion boundary. The observer drops all
but the newest satisfied descriptor, increments the coalescing-loss counter by
the number discarded, and emits only the newest interval with that observation
tick. It never fabricates distinct end ticks.

On an ordered one-generation reset, the clear command proves all old targets
completed. The observer coalesces them, attempts to emit the newest with a
`reset_observed` record type and a device tick sampled at reset observation, and
then adopts the new generation. Only older or unordered stale descriptors are
discarded and counted. A same-generation head that remains unsatisfied while its
queue is full increments `stuck_descriptor_head` once for that episode. It is
not discarded by an arbitrary wall-clock timeout because it may represent a
valid long program; later launches may lose profiler descriptors but application
dispatch remains unblocked. Reset/termination provides the bounded recovery
path.

### 6.5 Completed-record handoff

Use one compile-time 64-entry SPSC completed-record queue from the observer to
dispatch_s. It stores 64 eight-word records (2,048 bytes), uses monotonic
`uint32_t` producer/consumer indices, and has compile-time assertions that the
capacity is a power of two and at least twice the maximum 32 accepted
descriptors across eight streams. The record payload is written before its
producer index. dispatch_s performs exactly one bounded record service action at
each existing safe service point:

- check a register-space pending/publication signal first;
- if it is clear, return with no cache invalidation;
- if records are pending, issue at most the measured visibility operation per
  service batch;
- forward one record to the existing reserved profiler transport or count a
  downstream drop;
- never loop waiting for space.

Blackhole has no uncached L1 alias for this handoff, so the design assumes a
whole-L1 invalidation may be required to consume a TRISC-written payload. The
hard rules are: the empty check uses register space, no invalidation occurs when
the pending signal is clear, and invalidations per serviced record/batch have a
locked measured budget.

No profiler notification may be inserted between
`cq_noc_async_write_init_state` and `cq_noc_async_write_with_state` in the go
path. On Blackhole, `dispatch_s_noc_inline_dw_write` routes L1-destination writes
through `NCRISC_WR_CMD_BUF` even when another command buffer is requested; a
profiler NOC operation in that window can silently corrupt go-signal state.
Milestone 0 records the exact safe service points and Milestone 2 tests this
invariant.

### 6.6 Existing profiler ring

The reserved realtime-profiler core retains the existing bounded ring and host
transport. A full ring increments `device_ring_drops` and discards the incoming
interval. It never waits for host acknowledgement or free space.

TRISC0 stamps observer loss in the completed record, dispatch_s adds its own
launch/descriptor loss to the outbound mailbox copy, and the reserved BRISC adds
its cumulative ring loss to each accepted ring copy. A sentinel after pressure
therefore observes downstream loss even though TRISC0 cannot read reserved-core
state. The 32-bit wire aggregate is extended to 64 bits by the host.

The ring publishes record payload before its committed index. Schema and layout
changes participate in the firmware/JIT cache key and have compile-time size,
alignment, and offset assertions on both producer and consumer sides.

### 6.7 Generation and reset

The reset protocol is an ordered lifecycle transaction. In the supported
co-located topology dispatch_d clears the counter, publishes the shared-L1
generation and a 24-bit reset epoch in stream 8 scratch register 3, and only
then processes the ordered next-go notification that advances dispatch_s's
per-stream sync semaphore. The next descriptor cannot cross that semaphore.
dispatch_s and TRISC0 observe the reset epoch, invalidate L1, and adopt the new
generation before accepting/consuming new descriptors.

If the current command topology cannot order those steps, concurrent profiling
is inactive for that topology. A launch observed in the clear-to-generation
window must be rejected/count as reset loss; it must never produce a fabricated
interval from a stale target.

Tests exercise the actual `CLEAR_STREAM`/sub-device-manager lifecycle and an
adversarial launch in the clear-to-generation publication window. Direct L1
mutation remains fault injection and is not accepted as reset-path coverage.

### 6.8 Shutdown

Observer shutdown and terminal draining are bounded by item and cycle budgets.
Unconsumed descriptors/records are counted separately, and observer-stop
timeout is surfaced as a protocol error. Shutdown may synchronize device
lifecycle teardown, but it cannot wait indefinitely and it cannot conceal loss.

## 7. Continuous collection only

The clean implementation deliberately does not introduce the working
reference's watermark protocol, `FinishAndCollectProgramRealtimeProfiler`,
per-stream control slots, or large host collection registry. No current
production consumer uses that API; Sparse MLA profiling already consumes the
continuous callback.

Ordinary `Finish` emits zero realtime-profiler commands when no future,
separately approved collection feature exists. Stream-mask and sub-device-limit
validation happens during capability initialization, not inside `Finish`.

Deterministic profiler tests use the continuous path:

1. register a callback and snapshot host/device loss counters;
2. submit a known set of runtime IDs and record the expected publish-sequence
   count;
3. use ordinary queue completion only to establish application completion;
4. wait on the host test harness, with a finite test-only timeout, until the
   expected IDs/sequences arrive or a loss snapshot proves why they cannot;
5. compare terminal counter snapshots and fail on unexpected loss.

The host wait is test orchestration, never an operation-duration source. Loss
stress tests send a final sentinel after pressure is released so a fresh device
loss snapshot reaches the callback even when all stressed intervals were
dropped.

## 8. Host API and diagnostics

Preserve the shipped names and make additions explicit:

- `ProgramRealtimeRecord` retains its existing fields and adds command queue,
  dispatch stream, generation, successful sequence, schema/type, and cumulative
  source-loss fields;
- `ProgramRealtimeRecordBatch::dropped` continues to mean host callback-ring
  loss only;
- `ProgramRealtimeRecordBatch` adds a span of per-device
  `ProgramRealtimeProfilerDeviceLossSnapshot` values containing chip ID and the
  newest cumulative aggregate source loss in the batch;
- `RegisterProgramRealtimeProfilerCallback` and
  `UnregisterProgramRealtimeProfilerCallback` remain source-compatible;
- `IsProgramRealtimeProfilerActive()` remains and returns true when any managed
  device is active;
- `GetProgramRealtimeProfilerDeviceCapabilities()` is the single new query and
  returns `{chip_id, active, inactive_reason, detailed_loss_counts}` for
  initialized devices. The exact enum and structs are fixed by the Milestone 0
  protocol decision record.

`FinishAndCollectProgramRealtimeProfiler` and its collection-result structs are
not part of the clean implementation. They belong only to the working reference
and are never introduced on the pre-concurrent baseline.

Test-only saturation and fault injection live in test support rather than the
production manager API. Production code contains no outstanding collection
registry. Device-loss counters reach ordinary callbacks on every successful
record; the sentinel pattern covers all-records-dropped steady-state tests.
Terminal teardown counters are inspected only through the explicit diagnostic
query.

## 9. Performance contract

Milestone 0 records the clean baseline before implementation and locks numeric
budgets. Later milestones may not redefine the baseline or loosen a budget to
make a result pass.

At minimum, measure:

- dispatch_s cycles for a go command with profiler unsupported/disabled;
- dispatch_s incremental cycles for an accepted profiled launch;
- observer cycles while disabled, enabled-idle, with one active stream, and with
  all supported streams active;
- full-cache invalidation count and NOC transaction count in each state;
- observer completion-detection latency distribution and maximum under load;
- application device latency and throughput with profiling disabled/enabled;
- firmware binary and L1 deltas;
- sustained and burst interval throughput before counted loss;
- per-stream descriptor drop rate at the shortest supported program duration
  with at least two active streams, comparing descriptor depths two and four;
- source inspection that the observer is created only for eligible Blackhole
  configurations; no Wormhole build or runtime evidence is required.

Measurement method:

- all host-throughput arms interleaved in one balanced QuietBox session;
- each binary launched from its own source working directory with an isolated
  JIT cache; setting `TT_METAL_HOME` alone is insufficient because kernel
  lookup prefers the process working directory;
- warmup separated from measured iterations;
- at least 30 paired samples for noisy host-facing throughput;
- device-cycle measurements are primary;
- same-binary and cross-build host throughput are corroborating data for M1;
  they become gates only where their no-op control demonstrates sufficient
  resolution;
- a comment-only rebuild control must demonstrate that any proposed
  cross-build numeric gate is resolvable before that gate is used;
- report median, p95, range, and confidence interval;
- preserve raw output and exact commands.

Initial hard constraints, independent of statistical budgets:

- zero profiler-owned dispatch waits;
- zero per-launch cache invalidations or NOC operations when unsupported or
  disabled after initialization;
- zero all-stream scans while enabled-idle;
- bounded O(1) launch publication;
- bounded O(active streams) observer scan;
- zero unexplained or silent loss.

Milestone 0 adds numeric cycle, latency, code-size, and L1 ceilings based on the
clean baseline and hardware variance. Claude must approve those ceilings before
implementation begins.

## 10. Implementation milestones

Each milestone produces one well-rounded commit. A milestone is not committed
or published until its focused tests pass and Claude Opus returns exact
`APPROVE`. Review findings must be resolved within the milestone; they must not
expand scope into models, other architectures, or unrelated profiler work.

### Milestone 0: Freeze the contract and prove hardware ownership

Deliverables:

- Preserve `335393399b6` under a stable reference for comparison.
- Create a profiler-only implementation line from the exact pre-concurrent
  baseline `5fa57f6d18b`; inherit the existing transport/callback path but do not
  inherit `20e839e1` runtime changes.
- Map start issuance, completion counters, real reset paths, RISC ownership,
  cache visibility, and supported dispatch topology.
- Prove the depth-four, compile-time-sized SPSC descriptor and completed-record
  ownership model.
- Freeze the host-zero/device-ready/activation handshake and name the sole
  runtime writer of every shared field.
- Prove the register-space publication signal and local active-stream mechanism.
- Record the go-path NOC command-buffer hazard and exact safe service points.
- Define deterministic continuous-callback test scaffolding without a watermark.
- Add a reusable non-trace burst generator so later transport and loss tests do
  not depend on a replay mode that is intentionally unprofiled.
- Define the exact public record, loss-snapshot, capability structs, and
  compatibility behavior from Section 8.
- Measure the clean performance, firmware-size, and L1 baselines and lock
  numeric acceptance budgets.
- Write one short protocol decision record containing invariants, layouts,
  fence/cache rules, support gating, and rejected alternatives.
- Review the decision record and evidence with Claude Opus.

Exit criteria:

- Every state field has exactly one writer and named readers.
- Every producer/consumer edge has a proven publication order.
- All real completion-counter reset paths are known.
- Unsupported configurations are enumerated.
- Budgets and exact measurement commands are fixed.
- Claude returns exact `APPROVE`.

Commit: protocol decision, baseline evidence, and test scaffolding only.

### Milestone 1: Capability gating and protocol cleanup

Deliverables:

- Add strict Blackhole/single-CQ/topology capability checks.
- Add compile-time protocol layout and go-command field validation.
- Carry runtime program ID and completion contribution in one go-command path.
- Delete the program-ID FIFO append/pop/storage path as a dispatch deadlock fix.
- Publish start-only M1 records: preserve the device start timestamp and set
  the public end timestamp to zero. The inherited observer endpoint is for
  launch N-1 while the Go-carried runtime ID is for launch N; Milestone 2
  replaces it before intervals are exposed.
- Delete the legacy completion observer and its wait-loop timestamp sampling.
  Only profiled GO commands may capture a timestamp or notify the reserved
  core; M2 introduces the replacement observer.
- Latch disabled/unsupported state so it has no per-launch cache or NOC work.
- Publish activation in dependency order (remote address, then activation
  release) and latch late D2H-socket failure as terminal-disabled so a
  precompiled dispatch_s cannot add per-command polling.
- Keep dispatch telemetry independent of the realtime-profiler gate.
- Encode trace-captured go commands as unprofiled and prove replay emits zero
  descriptors while ordinary profiling resumes afterward.
- Add Blackhole negative tests for multi-CQ and unsupported topology, plus
  trace-command suppression and replay-resume coverage. Non-Blackhole hardware
  qualification is explicitly outside scope.
- Add a regression with more than 32 profiled programs while dispatch_s is
  stalled on a downstream semaphore; dispatch_d must continue making progress.
- Revisit the profiler-FIFO coverage exclusion documented near
  `test_prefetcher.cpp:1651` now that the spin is gone.

Exit criteria:

- There is no profiler-owned loop in go-command dispatch.
- Unsupported configurations fail before activation and never time out later.
- Existing callback delivery and runtime-ID/source correlation remain intact;
  M1 explicitly exposes no duration.
- The legacy observer is absent; M1's TRISC0 image is a stub. Wormhole
  build/runtime qualification is not required.
- Disabled GO-tail device cycles do not regress the exact clean baseline;
  the enabled full GO-tail increment is bounded and reported. Host timing is
  retained with its no-op resolution control and cannot override the primary
  device measurement.
- Claude returns exact `APPROVE`.

### Milestone 2: Minimal concurrent device path

Deliverables:

- Implement depth-four, compile-time-sized per-stream descriptor queues.
- Implement generation-aware completion targets.
- Implement register-space publication signaling, observer-local active-stream
  state, and the fixed 64-entry completed-record queue.
- Add the minimum correct Blackhole publication/cache operations.
- Add every loss/protocol counter named in Section 5.3.
- Exercise actual reset/reload firmware paths.

Focused tests:

- two streams complete in reverse submission order;
- back-to-back same-stream launches with another stream active;
- descriptor and completed-queue saturation never blocks dispatch;
- multiple descriptors satisfied in one scan produce one newest record and an
  exact coalescing-loss count;
- a stuck head is observable and never blocks application dispatch;
- completion-count wrap, actual reset generation, and an adversarial launch in
  the clear-to-generation window;
- publication ordering under repeated stress;
- no profiler NOC operation in the protected go-command stateful-write window;
- shortest-duration descriptor drop rate at depths two and four;
- disabled, idle, one-stream, and all-stream observer cycle measurements.

Exit criteria:

- Successful records correlate to the correct invocation and stream.
- Every injected loss is counted at the correct stage.
- There is no application wait on profiler state.
- Device timing contains no host substitution.
- Performance and footprint budgets pass.
- Claude returns exact `APPROVE`.

### Milestone 3: Existing transport and host loss visibility

Deliverables:

- Feed completed records into the existing reserved-core profiler ring.
- Make ring-full behavior drop-and-count.
- Extend the existing host parser/callback with schema, stream identity, publish
  sequence, and device-loss snapshot.
- Expose monotonic diagnostics to ordinary profiler consumers.
- Preserve existing callback/active-query compatibility and add only the
  explicit capability/loss fields defined in Section 8.
- Remove production-only test hooks, the watermark/collection protocol,
  `FinishAndCollectProgramRealtimeProfiler`, and oversized manager state.
- Make ordinary `Finish` emit zero realtime-profiler commands.

Focused tests:

- delayed/paused host drain;
- device-ring and callback-ring saturation;
- non-trace burst generation under each pressure mode;
- payload-before-index visibility and schema/cache-key mismatches;
- consumer detects every induced gap or loss delta;
- deterministic callback tests reach terminal success/loss without a watermark;
- sustained continuous profiling without explicit collection;
- ordinary `Finish` adds no profiler command or wait.

Exit criteria:

- Continuous callback users can distinguish lossless from lossy data.
- Host slowdown cannot block firmware/application progress.
- Firmware and host state remain statically bounded.
- Claude returns exact `APPROVE`.

### Milestone 4: Qualification and replacement decision

Deliverables:

- Run the full functional, reset, overload, and performance matrix on QuietBox.
- Compare the clean implementation against both the pre-concurrent baseline and
  `335393399b6` using identical commands/builds.
- Verify no D2H tensor fallback or host-duration substitution exists.
- Audit all loops, waits, cache invalidations, and NOC operations reachable from
  profiler code.
- Document the timing error distribution, loss behavior, support matrix,
  firmware/L1 cost, and raw evidence.
- Inspect the Blackhole-only observer-creation gate; no Wormhole build or
  runtime qualification is required.
- Obtain final Claude Opus approval of the complete profiler diff and evidence.

Exit criteria:

- All locked correctness and performance budgets pass.
- No profiler-owned dispatch wait exists.
- Unsupported configurations fail explicitly.
- Continuous consumers observe device loss.
- The implementation is smaller and has lower enabled-idle/hot-path cost than
  `335393399b6`, or any exception has measured evidence and explicit approval.
- Claude returns exact `APPROVE`.

Commit qualification tests and maintained architecture documentation as the
final profiler milestone. Milestone process notes and raw logs remain artifacts
rather than permanent runtime documentation unless they are needed to maintain
the feature.

## 11. Review workflow

The Sol high-effort review of `335393399b6` is an input to this plan. Its
mandatory findings are carried into the milestones: remove the blocking FIFO,
gate multi-CQ/topology, test real reset, correct “exact” terminology, expose
device loss, and eliminate unnecessary idle scanning/cache work.

The complete revised plan received exact `APPROVE` from Claude Opus at high
effort after code-backed verification of those findings. Milestone 0 fixes the
loss-snapshot encoding and descriptor-publication order in the protocol decision
record before runtime implementation begins.

Claude Opus reviews exactly once at each milestone boundary after tests and
evidence are ready:

```bash
claude --dangerously-skip-permissions --model opus --effort high --print \
  "Review milestone <N> of the clean-room Blackhole concurrent realtime \
  profiler. Verify the stated contract, supported topology, single-writer \
  ownership, bounded/nonblocking behavior, device-only timing semantics, loss \
  visibility, hot-path cost, tests, and evidence. Check that no model, \
  Wormhole feature support, multi-CQ, D2H tensor fallback, or host-duration \
  substitution entered scope. Wormhole build/runtime qualification is explicitly \
  excluded by product-owner direction. Respond with \
  exactly APPROVE or CHANGES REQUIRED followed by concrete \
  findings."
```

One review gate per milestone prevents perpetual design churn. A reviewer may
block a milestone for a correctness, realtime, support, or evidence violation.
Feature requests outside the approved contract are recorded as follow-ups and
do not trigger redesign of the milestone.

## 12. Commit and branch strategy

- Keep `335393399b6` reachable as the working-reference implementation.
- Develop the profiler from exact pre-concurrent baseline `5fa57f6d18b` on
  `pjosipovic/realtime-profiler-clean-room`; do not mix model or operation
  changes into its commits.
- Produce one commit for each approved milestone.
- Publish after every milestone using normal non-destructive pushes.
- Do not rewrite an approved milestone unless a later correctness fix genuinely
  belongs there; if history is rewritten, use `--force-with-lease` and record
  old/new tips.
- Integrate the final profiler commit series into the Sparse MLA branch only
  after Milestone 4 passes.

## 13. Definition of perfect for this scope

The implementation is complete when:

- overlapping Blackhole sub-device programs produce correctly correlated raw
  device intervals;
- the timing contract and its device-side observation error are explicit;
- no profiler state can block or backpressure application dispatch;
- unsupported configurations never partially activate;
- disabled, idle, and active costs are bounded, measured, and within locked
  budgets;
- all buffers are bounded and all loss is visible to normal consumers;
- the real counter-reset lifecycle and wrap behavior are tested;
- normal realtime streaming remains the primary path;
- deterministic tests use the continuous stream without a watermark protocol;
- no D2H tensor fallback or host-duration substitution exists;
- the maintained implementation is materially smaller and simpler than the
  working reference while retaining its proven concurrent behavior;
- every milestone and the final design receive Claude Opus `APPROVE`.
