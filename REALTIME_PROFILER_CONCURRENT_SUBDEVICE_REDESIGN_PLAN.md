# Concurrent Sub-Device Profiler Plan

**Branch:** `pjosipovic/sparse-mla-topk-kv-gather-overlap`
**Scope:** Blackhole profiler runtime only
**Status:** Milestones 0 through 3 complete

## 1. Goal

Extend the lightweight, always-on device profiler so it can report accurate
device-time intervals for operations executing concurrently on different
Blackhole sub-devices.

The new path must preserve the profiler's core design goals:

- device-side timestamps, not host wall-clock timing;
- no device-to-host tensor fallback;
- no application-thread wait for an individual profiler record;
- bounded device memory, firmware work, and host work;
- explicit, observable loss instead of silent record corruption or blocking;
- negligible effect on unprofiled execution and a measured, bounded effect when
  lightweight profiling is enabled.

This phase delivers a general profiler capability. It does not integrate or
tune any model operation.

## 2. Scope

### In scope

- Blackhole single-command-queue execution.
- Concurrent operations assigned to distinct sub-devices.
- Device start and completion timestamps associated with the correct program
  and sub-device stream.
- Nonblocking publication from firmware to the existing host profiler path.
- Ordered collection with a device-produced completion watermark.
- Explicit overflow and dropped-record counters.
- A host API that can wait for or query a completed profiler batch without
  using host timing as the measurement.
- Runtime and profiler tests using synthetic concurrent programs.
- Lightweight-profiler overhead, memory, and code-size measurements.

### Out of scope

- Sparse MLA integration.
- `topk_large_indices`, `high_bw_all_gather`, or any other operation-specific
  changes.
- Sub-device core allocation policy or specific core splits.
- Model performance tests or model speedup claims.
- Wormhole support or compatibility for this new feature.
- Quasar support.
- Multi-command-queue support.
- Non-MMIO Galaxy relay and cross-chip clock calibration.
- Trace replay support for concurrent profiler intervals.
- Redesign of the existing host Tracy output format beyond the minimum metadata
  required to identify concurrent device intervals.

Wormhole is not a deliverable for this feature. Blackhole implementation must
be architecture-guarded, and no milestone is blocked on Wormhole behavior or
test parity.

## 3. Required behavior

### 3.1 Interval identity

Every successfully collected interval must identify:

- device/chip;
- command queue;
- program invocation;
- sub-device or dispatch stream;
- start timestamp;
- completion timestamp;
- record type and schema version.

The identity must distinguish two programs that overlap in time on different
sub-devices. Host submission order alone must not be used to infer completion
order.

### 3.2 Device-time semantics

- Start is captured at the existing device-side program-start point, or at a
  replacement point proven to represent the same event.
- End is captured after the program's actual completion condition is observed
  for its stream.
- Host enqueue, callback, drain, and synchronization timestamps are diagnostic
  only and are never reported as operation duration.
- Timestamp-domain assumptions must be documented and tested on the supported
  Blackhole configuration.

### 3.3 Nonblocking semantics

The application dispatch path must not wait for the host to acknowledge or
drain an individual profiler record.

All producer actions must have a finite bound. When capacity is exhausted, the
profiler must:

1. preserve execution correctness;
2. avoid waiting for host progress;
3. increment an observable loss counter;
4. allow later records to resume once capacity returns.

The implementation must not turn ordinary `Finish` or queue synchronization
into a hidden per-record profiler drain. An explicit profiler collection API
may wait for a requested completion watermark.

### 3.4 Completion watermark

The device must publish an ordered watermark that means all successfully
published interval records through sequence `N` are available to the host, or
that loss affecting the requested range has been reported.

The host must not decide that a batch is complete merely because a ring is
temporarily empty. The batch result must contain:

- requested watermark;
- observed watermark;
- record count;
- dropped-record count;
- overflow/protocol error state.

### 3.5 Lightweight behavior

When profiling is disabled, no new per-program wait, NOC transaction, or
steady-state polling may be added.

When lightweight profiling is enabled:

- producer work is constant and bounded per program;
- consumer work is proportional to published records;
- buffers are statically bounded;
- overflow is counted rather than backpressured into application execution;
- added firmware code and L1 usage remain within measured Blackhole budgets.

## 4. Design decisions to resolve before implementation

Milestone 0 must answer these questions from code inspection and focused
experiments. The plan intentionally does not preselect byte layouts or fence
sequences before those results exist.

1. **Completion source:** Which Blackhole firmware agent has authoritative
   knowledge of program completion for each sub-device stream?
2. **Start/end correlation:** What stable invocation identifier is available at
   both the start publisher and completion observer?
3. **Producer ownership:** Is the interval queue single-producer, or does it
   need multiple-producer arbitration?
4. **Visibility:** What is the minimum ordering/fence sequence needed between
   the publishing RISC, the collector RISC, and host-visible transport?
5. **Capacity:** What queue depth absorbs expected bursts without increasing
   the existing reserved profiler footprint beyond the agreed budget?
6. **Watermark ordering:** At what point can a watermark safely advance without
   overtaking a record still in another staging buffer?
7. **Loss accounting:** How are producer overflow, collector overflow, and host
   transport loss distinguished and reported?
8. **Lifecycle:** Which component initializes and resets sequence numbers,
   buffers, counters, and capability state for the supported Blackhole route?

The output is a short decision record containing the selected design, rejected
alternatives, measurements, and exact invariants. Claude Opus must approve that
record before Milestone 1 begins.

## 5. Proposed architecture

This is the working architecture to validate in Milestone 0:

1. Dispatch publishes a small start descriptor containing invocation and stream
   identity plus the device start tick.
2. The authoritative Blackhole completion observer matches that descriptor to
   the corresponding stream completion and captures the end tick.
3. A bounded device queue carries completed interval records to the existing
   profiler transport.
4. A monotonically increasing publish sequence orders successfully accepted
   interval records.
5. A watermark is emitted only after all accepted records through its sequence
   are visible to the downstream collector.
6. Capacity failures increment stage-specific counters and do not stall program
   execution.
7. The host receiver collects records and exposes watermark-based completion to
   tests and callers.

Milestone 0 may replace this architecture if evidence shows a simpler design
satisfies all invariants.

## 6. Milestones

### Milestone 0: Prove the protocol

Deliverables:

- Map current Blackhole start, completion, and profiler transport paths.
- Identify all current waits introduced by commit
  `20e839e1ff9032573690c031751ad58b5e967fed`.
- Prove the completion target used for each supported sub-device stream.
- Run focused visibility and ordering experiments for candidate RISC handoffs.
- Measure current L1, firmware code-size, and profiler overhead baselines.
- Produce the decision record specified in Section 4.
- Obtain exact `APPROVE` from Claude Opus for the decision record.

Exit criteria:

- No unresolved producer, completion, ordering, or lifecycle ownership.
- Proposed producer and collector actions are demonstrably bounded.
- Memory and code-size budgets are stated numerically.
- Implementation files and tests are identified.

Commit this milestone as documentation and focused experimental tests only.

### Milestone 1: Device publication path

Deliverables:

- Add Blackhole-only interval descriptors and capability gating.
- Capture start and completion ticks with stable invocation/stream identity.
- Add bounded record publication, sequence allocation, and stage-specific drop
  counters.
- Remove Blackhole per-record ACK/drain waits replaced by the new protocol.
- Add firmware/unit tests for ordering, wraparound, full-buffer behavior, and
  concurrent completion order.
- Measure firmware code-size, L1 use, and disabled/enabled micro-overhead.
- Obtain Claude Opus review and exact `APPROVE` for the milestone diff and test
  evidence.

Exit criteria:

- Two synthetic programs can complete out of submission order and produce the
  correct device intervals.
- Buffer exhaustion never stalls application execution.
- Every intentional drop is visible in a counter.
- Disabled-path regression gates pass.

Commit and publish after approval.

### Milestone 2: Watermark and host collection API

Deliverables:

- Add ordered device watermarks that cannot overtake accepted records.
- Extend host parsing for concurrent interval identity and loss metadata.
- Add a collection result/API that waits for a requested device watermark.
- Keep timing calculations based exclusively on device ticks.
- Add tests for delayed host drain, empty-ring races, overflow, reset, sequence
  wrap, and collection timeout/error reporting.
- Obtain Claude Opus review and exact `APPROVE` for the milestone diff and test
  evidence.

Exit criteria:

- The host never reports a batch complete based only on ring emptiness.
- A successful result proves all accepted records through the requested
  watermark were collected.
- A lossy result identifies the affected batch and reports nonzero loss.
- No D2H tensor fallback or host-duration substitution exists.

Commit and publish after approval.

### Milestone 3: End-to-end profiler qualification

Deliverables:

- Add a Blackhole synthetic concurrent-sub-device end-to-end test.
- Validate overlapping and non-overlapping intervals from device timestamps.
- Stress delayed host collection and bursty completion.
- Compare disabled and lightweight-enabled overhead against the Milestone 0
  baseline.
- Archive device interval output, loss counters, test commands, hardware/firmware
  revision, and statistics.
- Obtain final Claude Opus review and exact `APPROVE` for the complete feature.

Exit criteria:

- Device intervals correctly show concurrency without using host timing.
- Repeated stress runs have zero unexplained record loss or protocol errors.
- Overflow injection produces counted loss without application stalls.
- Disabled overhead, enabled overhead, L1, and code-size stay within the
  Milestone 0 budgets.
- All supported Blackhole profiler and runtime tests pass.

Commit and publish after approval.

## 7. Testing strategy

### Functional tests

- Two sub-devices with deliberately reversed completion order.
- Back-to-back programs on one stream while another stream remains active.
- Identical program binaries with distinct invocation identifiers.
- Timestamp and sequence wrap boundaries.
- Buffer-full injection at each bounded stage.
- Delayed and temporarily paused host receiver.
- Profiler initialization and supported reset/reopen lifecycle.

### Correctness assertions

- `end_tick >= start_tick` under the documented wrap rule.
- Each accepted invocation produces at most one completed interval.
- Every completed interval maps to exactly one start descriptor.
- Published sequence values are unique and ordered.
- Watermark `N` never becomes visible before every accepted record through `N`.
- Drops are monotonic, stage-specific, and included in the collection result.

### Performance measurements

Use paired runs on the same Blackhole system and report distributions, not one
sample:

- profiler disabled baseline;
- existing lightweight profiler baseline;
- new concurrent profiler enabled with no overlap;
- new concurrent profiler enabled with overlapping synthetic programs;
- delayed-host stress.

Measure:

- device program duration;
- host submission throughput as an overhead signal only;
- firmware cycles spent per published record;
- L1 and firmware binary delta;
- records per second;
- overflow/loss counts.

Milestone 0 sets numeric acceptance thresholds from baseline noise. Later
milestones may not weaken those thresholds without a documented and reviewed
reason.

## 8. Review and commit workflow

Each milestone follows the same gate:

1. Implement only that milestone.
2. Run its focused tests and collect the required measurements.
3. Review the diff for unrelated changes and accidental model integration.
4. Run Claude Opus locally:

   ```bash
   claude --dangerously-skip-permissions --model opus --effort high --print \
     "Review milestone <N> of the Blackhole concurrent sub-device profiler. \
     Check correctness, bounded/nonblocking behavior, device-time semantics, \
     lightweight-profiler overhead, tests, and scope. Respond with exactly \
     APPROVE or CHANGES REQUIRED followed by concrete findings."
   ```

5. Address concrete findings within milestone scope.
6. Require exact `APPROVE` before committing.
7. Create one well-rounded milestone commit and push it to the published branch.

Claude review does not replace tests or measurements. Review iterations should
not expand the feature into model integration, Wormhole compatibility, remote
Galaxy transport, or unrelated profiler redesign. Any newly discovered work in
those areas is recorded as a follow-up.

## 9. Definition of done

The profiler feature is complete when:

- Blackhole concurrent sub-device operations produce correctly correlated
  device start/end intervals;
- record publication is bounded and does not wait for host progress;
- watermark-based collection detects incomplete or lossy batches;
- all loss is explicit and no D2H fallback exists;
- disabled and enabled overhead remain within measured budgets;
- functional, stress, lifecycle, and performance tests pass;
- the final implementation and evidence receive Claude Opus `APPROVE`;
- milestone commits are published on the feature branch.

Model integration, operation core allocation, Galaxy relay, and support for
other architectures are separate projects after this profiler capability is
qualified.
