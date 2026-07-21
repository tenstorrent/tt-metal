# 8x1 sparse-MLA all-gather bandwidth plan

> Current qualification status (2026-07-21): the receiver-tail accounting fix
> closes the final correctness gate. The env-free bank-fanout and depth-4
> terminal-offload policy passes full 65,536-row/device BF16 and scaled-FP8
> output checks and reaches 92.465 and 92.026 GB/s respectively on the
> Galaxy-compatible link configuration. The remaining product gate is four
> concurrent SP=8 rings on a physical Galaxy.

## Objective

Move the native all-gather for the production long sparse-MLA geometry closer
to the enabled 100 GB/s per-rank effective-receive roofline on an eight-device
Blackhole ring.  Optimize the isolated CCL first, then retain only changes that
improve the model-compatible one-row-per-page layout.

The canonical workload is:

- topology: `8x1`, `FABRIC_1D_RING`;
- rows per rank: `64,640`;
- global rows: `517,120` (`512,000` cached plus `5,120` live);
- BF16 page: `1,152 B`;
- scaled-FP8 page: `704 B`;
- persistent interleaved DRAM output;
- release firmware, never a `--dev` build.

`effective_receive_bw` is the aligned tensor bytes received by one rank from
the other seven ranks divided by the all-gather device-program duration.  It is
not a measurement of raw Ethernet wire traffic.

## Baseline and target

| Format and path | Median | Effective receive BW | 100 GB/s latency | Required latency reduction |
| --- | ---: | ---: | ---: | ---: |
| BF16, native direct | 8.600 ms | 60.612 GB/s | 5.213 ms | 39.4% |
| scaled FP8, receiver L1 | 6.319 ms | 50.408 GB/s | 3.185 ms | 49.6% |

The enabled roofline is four aggregate ingress links per rank at the CCL's
modelled 25 GB/s per link.  Exact 100 GB/s would require ideal payload duty
cycle, so the first practical milestone is 80 GB/s and the stretch milestone
is 90 GB/s without a p90 or correctness regression.

For scaled FP8 these milestones correspond to:

| Milestone | Maximum median |
| --- | ---: |
| 60 GB/s | 5.309 ms |
| 70 GB/s | 4.551 ms |
| 80 GB/s | 3.982 ms |
| 90 GB/s | 3.539 ms |
| 100 GB/s | 3.185 ms |

## Invariants

Every retained experiment must preserve:

1. The public `ttnn.all_gather` contract and the generic direct fallback.
2. Correct full gathers, tails, partial extents, and batch slices.
3. Persistent/preallocated output support.
4. Ordinary L1 payload slots and L1-small control semaphores.
5. Monotonic produced/consumed sequence values and one credit per destination.
6. No source slot reuse before its Fabric payload has completed and every
   destination has acknowledged consumption.
7. Sparse MLA's one-token-row-per-page cache layout unless a separately proven
   downstream-compatible layout is introduced.
8. Release firmware.  Use `scripts/run_safe_pytest.sh`; do not use `--dev`.

Any hang is diagnosed with `tt-triage` before changing synchronization.  A
performance change is not retained merely because it happens to avoid a hang.

## Measurement discipline

- Compile and warm before measurement.
- Use seven steady-state device-profiler samples.
- Report median, minimum, p90, all samples, path, packet rows, slots, credit
  mode, drain RISCs, and effective receive bandwidth.
- Use A/B/A when run-to-run drift could explain a result.
- Enable device attribution only for stage diagnosis, then remeasure cleanly.
- Change one mechanism at a time.
- Retention gate: at least 3% median improvement, no material p90 regression,
  exact output correctness, and repeated stability.

Canonical isolated measurement (the geometry and production policy are fixed by
the test; only the standard device profiler is enabled externally):

```bash
TT_METAL_DEVICE_PROFILER=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k sparse_mla_row_perf -q -s
```

After an attribution run exits safely:

```bash
scripts/analyze_all_gather_attribution.py --samples 7
```

## Phase 1: exact 8x1 stage attribution

Measure the production-size scaled-FP8 path in these modes:

| Mode | Environment | Question |
| --- | --- | --- |
| Direct | `RECEIVER_L1_MODE=force_direct` | What does the established path achieve? |
| L1 sink | `RECEIVER_L1_MODE=force_receiver`, `RECEIVER_STAGE_MODE=l1_sink` | What is the producer plus Fabric-ingress ceiling without DRAM drain? |
| Drain only | `RECEIVER_STAGE_MODE=drain_only` | How fast can prefilled receiver L1 be drained to the real output? |
| Combined | `RECEIVER_STAGE_MODE=combined` | How much overlap survives in the complete protocol? |

Run BF16 after scaled FP8 so dtype-specific bottlenecks are not conflated.

Decision rules:

- L1 sink below 80 GB/s: optimize sender/Fabric issue before receiver fan-out.
- Fast L1 sink but slow drain-only: optimize receiver bank/NoC scheduling.
- Both micro-stages fast but combined slow: remove credit or buffer-reuse
  serialization.
- Direct faster than receiver for BF16: keep dtype-specific selection while
  diagnosing BF16 separately.

## Phase 2: proactive grouped-credit pipeline

Exact 8x1 attribution changed the next implementation priority.  The receiver
can drain FP8 above 100 GB/s in isolation, while L1-sink is near 54 GB/s and
the sender spends multiple milliseconds waiting for credit epochs.  The
existing window protocol creates an all-destination round trip whenever all
L1 slots have been consumed.  The existing per-slot protocol performs even
worse because it synchronously pays that round trip for every slot.

Add a third, test-controlled credit mode that:

1. divides a slot ring into consumption groups;
2. has the receiver publish each group as soon as both drain RISCs have
   consumed it;
3. lets the mirrored sender worker observe published groups without blocking
   and immediately proxy their Fabric credit increments;
4. waits for aggregate remote credit only when a slot is about to be reused;
5. therefore overlaps a group's credit round trip with transmission and drain
   of the remaining slot groups.

Sweep group sizes `1`, `2`, `4`, `6`, and a full slot window.  Group size must
not alter slot addresses or produced sequence numbers.  For dual-RISC drain,
both RISCs must complete a group before it is published.  The final completion
path must still wait for every payload and drain, even when the last group's
credits are unnecessary for reuse.

TDD order:

1. Host/reference tests for wrap and reclaim sequences.
2. One-RISC L1-sink small and production-size cases.
3. Dual-RISC combined tails and exact output correctness.
4. Exact 8x1 FP8 combined A/B/A.
5. BF16 regression control; automatic BF16 remains direct unless a receiver
   candidate beats it independently.

## Phase 3: multi-batch Fabric completion window

The sender currently calls `fabric.async_writes_flushed()` after every transport
batch.  Receiver slots and window credits therefore do not necessarily create a
deep source-to-Fabric pipeline.

Add a test-only policy with completion windows `1`, `2`, `3`, `6`, and `auto`.
Window `1` must reproduce the current implementation.  A larger window may
defer Fabric completion only while every source L1 address remains unique and
cannot be overwritten by the reader, the opposite-direction writer, or CB
reuse.  Always flush:

- before wrapping to an in-flight CB/L1 slot;
- before returning a receiver slot credit that could enable reuse;
- before completion notification and connection close;
- at a partial final window.

TDD order:

1. Host parser and policy tests.
2. Small one-page, one-tail, and wraparound cases.
3. Repeated 8x1 correctness with windows `1`, `2`, and `3`.
4. Production-size L1-sink A/B to isolate transport effect.
5. Production-size combined A/B/A.

If the Fabric API or NOC command queue cannot safely retain multiple source
buffers, record that structural limit and remove the experimental policy.

## Phase 4: source CB and transaction depth

The source CB currently has depth three.  Once Fabric completion is windowed,
sweep CB depths `3`, `4`, `6`, and `8`, limited by L1 capacity.  Match the NOC
transaction-ID ring and source buffer lifetime to the selected depth.

Measure separately:

- source DRAM issue and wait cycles;
- CB reserve/wait cycles;
- Fabric issue/completion cycles;
- receiver produced wait cycles.

Retain the smallest depth that reaches the plateau; extra L1 use without a
measured gain is a regression risk.

## Phase 5: route-stream decomposition

Only enter this phase if L1-sink bandwidth remains below 80 GB/s after Phases 2
and 3.  Today one worker per routing plane manages both directions and even-ring
alternate routes.  Determine whether a single data-movement RISC is unable to
issue enough work to keep both route directions busy.

Prototype independent route streams while staging each source batch once in
L1.  Give every stream disjoint packet headers, command state, and completion
accounting.  Do not duplicate source DRAM reads unless attribution proves that
L1 fan-out is slower.

Correctness gates include the opposite ring node, where route selection changes,
and sequence wrap across more batches than available source and receiver slots.

## Phase 6: bank-owned source and receiver schedules

Revisit the existing diagnostic bank-owned schedule only after transport
serialization is removed.  The earlier prototype reduced source and receiver
command issue substantially but lost its gain in the sender pipeline.

The production candidate should:

- statically assign disjoint DRAM banks to source workers;
- read already-contiguous bank runs directly into final transport order;
- avoid an inline per-payload permutation on a single RISC;
- target receiver workers/RISCs with matching disjoint output banks;
- preserve logical tensor order through address generation rather than a model
  side data shuffle.

Extend the diagnostic path from four to eight devices only after the exact 8x1
mapping proof passes.  Add receiver cores only if L1-sink is faster than
combined and drain-only attribution identifies receiver issue as the gap.

## Phase 7: layout co-design, only if necessary

If per-row destination commands remain the limiter, evaluate a DRAM-sharded or
bank-major output that sparse SDPA can consume directly.  This is a model and
kernel contract change, not an internal CCL tuning knob.  It must prove:

- no extra conversion or reshape dispatch;
- direct indexed access from sparse SDPA;
- unchanged cache capacity and numerical results;
- end-to-end improvement beyond the native interleaved-output path.

## Final verification

Before retaining the final policy:

1. Focused all-gather parser and correctness matrix.
2. Repeated exact 8x1 FP8 and BF16 isolated measurements.
3. Sparse MLA functional tests through `scripts/run_safe_pytest.sh`.
4. Sparse MLA perf warm/cold/long A/B.
5. `./build_metal.sh --release`.
6. QuietBox and Galaxy ring validation when those systems are available.
7. Update `sparse_mla_all_gather_perf.md` with clean final numbers and clearly
   distinguish effective receive bandwidth from raw Ethernet bandwidth.

## Work log

### 2026-07-19: plan started

- Created this execution plan before making pipeline changes.
- Recorded the exact production geometry and current BF16/FP8 baselines.
- First action: collect exact 8x1 direct/L1-sink/drain-only/combined attribution.
- No performance implementation has been changed in this phase yet.

### 2026-07-19: exact 8x1 stage attribution

All runs used 64,640 rows/rank, seven samples, the 8x1 ring, persistent DRAM
output, L1-small controls, release firmware, and the safe pytest runner.

Clean stage timings:

| Format | Path/stage | Median | Min / p90 | Effective receive BW |
| --- | --- | ---: | ---: | ---: |
| scaled FP8 | direct | 7.642 ms | 7.640 / 7.648 ms | 41.682 GB/s |
| scaled FP8 | receiver L1-sink, one RISC | 5.898 ms | 5.880 / 5.944 ms | 54.012 GB/s |
| scaled FP8 | receiver drain-only, two RISCs | 2.900 ms | 2.898 / 3.954 ms | 109.827 GB/s |
| scaled FP8 | receiver combined, two RISCs | 6.318 ms | 6.308 / 6.361 ms | 50.423 GB/s |
| BF16 | direct | 8.600 ms | 8.589 / 8.625 ms | 60.612 GB/s |
| BF16 | receiver L1-sink, one RISC | 9.523 ms | 9.484 / 9.551 ms | 54.738 GB/s |
| BF16 | receiver drain-only, one RISC | 6.287 ms | 6.287 / 6.287 ms | 82.916 GB/s |
| BF16 | receiver combined, one RISC | 10.830 ms | 10.826 / 10.846 ms | 48.132 GB/s |

The first clean FP8 drain-only sample was a 3.954-ms outlier; all remaining
samples were 2.898--2.901 ms.  It remains a diagnostic stage, not a correctness
result, because it drains unspecified prefilled L1 contents.

FP8 combined attribution with the device profiler enabled reported:

- 1,616 Fabric payloads and 32,320 source reads per directional worker;
- reader/writer credit waits of 4.288/4.150 ms;
- writer Fabric issue/flush of 0.333/0.218 ms;
- receiver drain issue of 4.018 ms and produced wait of 2.756 ms;
- 134 credit commands per directional worker, matching approximately one
  global credit epoch per full receiver slot ring.

FP8 L1-sink attribution removed all destination writes but still spent 3.245
ms reader-side and 3.094 ms writer-side in credit waits; receiver produced wait
was 5.743 ms.  The clean L1-sink ceiling is therefore only 54.0 GB/s.

The existing synchronous per-slot credit control was also measured in clean
FP8 L1-sink mode: 9.940 ms / 32.047 GB/s.  It is rejected as a performance
candidate but retained as a correctness control.

Decision: do not begin with more receiver drain cores or the end-of-batch
Fabric flush window.  Drain-only already exceeds the target and attributed
Fabric flush is too small to explain the gap.  Implement proactive grouped
credits so round-trip latency can overlap useful work.  Revisit Fabric flush
windowing only after the credit stall is removed or substantially reduced.

### 2026-07-19: proactive grouped-credit implementation, first result

- Added the `pipelined` receiver-credit policy, with its group size selected
  internally from the validated schedule.
- The receiver publishes one consumed sequence per completed group; source
  workers proxy every ready group to all destinations without waiting for slot
  reuse.  A source blocks only when it reaches a slot whose group credit has
  not completed globally.
- Added reference schedule tests for partial final groups and slot reuse.  All
  five formula cases passed.
- `./build_metal.sh --release` passed.
- A 4x2 ND-sharded sparse-KV correctness test passed with two-batch groups and
  a persistent interleaved DRAM output.
- The first exact 8x1 scaled-FP8 L1-sink measurement used 12 slots and the
  automatic four-batch group: 5.596 ms median, 5.579/5.599 ms min/p90, and
  56.928 GB/s.  This is 5.4% faster than the 5.898-ms / 54.012-GB/s window
  baseline, but still far from the target.  Next action: sweep group sizes to
  distinguish remaining credit latency from source/Fabric throughput.

### 2026-07-19: grouped-credit sweep and post-change attribution

Exact 8x1 scaled-FP8 L1-sink sweep with 12 slots:

| Credit group | Median | Effective receive BW |
| ---: | ---: | ---: |
| 1 batch | 7.027 ms | 45.333 GB/s |
| 2 batches | 6.077 ms | 52.421 GB/s |
| 4 batches | 5.596 ms | 56.928 GB/s |
| 6 batches | 5.500 ms | 57.918 GB/s |
| 8 batches | 5.658 ms | 56.297 GB/s |

Group 6 is the current command-overhead/credit-lead optimum.  The combined
two-RISC path passed correctness and measured 5.600 ms / 56.878 GB/s, a 12.8%
bandwidth improvement over the 6.318-ms / 50.423-GB/s window baseline.  Its
runtime is now within 0.100 ms of L1-sink, so receiver drain is largely hidden
but both producer and consumer still sit well below the target.

The first detailed-attribution run exposed an initialization race: a receiver
could publish ready before its sender initialized the local epoch.  Tt-triage
showed one sender at its local init wait, peers at grouped-credit waits, and
receivers waiting for that source's produced sequence.  Added an explicit
sender-to-receiver start handshake using the otherwise-unused local-source
produced semaphore.  A first version incorrectly reset the global barrier and
lost early remote increments; a second tt-triage capture identified the
receiver payload waits caused by that reset.  Retaining the monotonic barrier
epoch and only adding the local handshake passes dual-RISC correctness and the
previously hanging attribution run.

Three-sample group-6 combined attribution reported:

- reader/writer critical totals: 5.345/5.413 ms;
- reader/writer credit work: 2.997/2.627 ms;
- source/local issue: 1.729/1.749 ms for 32,320 commands per worker;
- Fabric issue/flush: 0.329/0.258 ms;
- receiver critical total: 5.477 ms;
- receiver drain issue: 4.015 ms for 113,120 commands on the critical RISC;
- receiver produced wait: 1.446 ms.

Decision: the 14-KiB payload itself is not the command-count problem, but the
704-byte DRAM source/local/receiver operations are.  Test the existing
bank-owned coalescing schedule next because it can collapse physical-bank runs
without reducing Fabric payload size.  Credit windowing alone cannot close the
remaining 2.4-ms gap to the 100-GB/s target.

### 2026-07-19: bank-owned coalescing and directional sequence fix

- Extended the bounded diagnostic bank-owned schedule to the exact 8x1 ring.
  Coalescing collapses each 20-row physical-bank run into one source read,
  local write, and receiver write while keeping the 14,080-byte Fabric packet.
- Corrected the FP8 correctness oracle to compare the gather with the values
  actually quantized into the input tensor.  The old pre-quantization reference
  could report conversion error as a CCL error.
- Found a real even-ring correctness bug at the opposite rank.  Alternate route
  balancing assigns consecutive opposite-rank batches to different sender
  workers, but one cumulative produced sequence allowed batch N+1 on one
  direction to satisfy the receiver's wait for batch N on the other direction.
- Rejected a per-slot ready-trailer prototype after it hung.  Tt-triage showed
  one receiver waiting for a missing slot-ready value, its senders waiting for
  credit, and the remaining ranks at all-gather completion.  The hang also
  reproduced with one drain RISC, ruling out dual-RISC drain synchronization.
- Retained separate forward/reader and backward/writer produced sequences for
  the bank-owned path.  The receiver selects the sequence by route; for the
  opposite rank it selects reader on even batches and writer on odd batches.
  Generic routing aliases both directions to its original sequence and does
  not allocate the additional L1-small controls.
- The final route-balanced implementation passed five repeated BF16 and five
  repeated scaled-FP8 8x1 exact-output tests.  The release build passed.

The coalescing mechanism works mechanically but did not improve the full
pipeline.  With group-6 credits, the exact scaled-FP8 result was approximately
5.70 ms / 55.8 GB/s versus 5.60 ms / 56.9 GB/s without all three coalescing
stages.  Fresh attribution explains why:

| Critical worker metric | Time / count |
| --- | ---: |
| reader total | 5.336 ms |
| reader credit wait/work | 3.475 ms |
| reader source issue | 0.145 ms / 1,616 commands |
| writer total | 5.526 ms |
| writer credit wait/work | 3.471 ms |
| writer local issue | 0.919 ms / 1,616 commands |
| receiver total | 5.489 ms |
| receiver drain issue + flush | 1.130 ms / 5,656 commands |
| receiver produced wait | 4.356 ms / 5,656 waits |

Compared with the uncoalesced attribution, source commands fell 20x and
receiver commands fell 20x.  The saved issue time became payload starvation:
sender credit waits and receiver produced waits now dominate almost the entire
critical path.  More receiver write cores alone cannot recover this gap.

### 2026-07-19: coalesced credit-window and packet-size sweeps

The exact 8x1 scaled-FP8 all-coalesced credit-group sweep used 20-row bank
runs, 12 automatic slots, and three samples:

| Credit group | Median | Effective receive BW |
| ---: | ---: | ---: |
| 1 | 7.160 ms | 44.490 GB/s |
| 2 | 6.158 ms | 51.729 GB/s |
| 3 | 5.856 ms | 54.394 GB/s |
| 4 | 5.707 ms | 55.814 GB/s |
| 6 | 5.717 ms | 55.717 GB/s |
| 8 | 5.910 ms | 53.896 GB/s |

Groups four through six are the plateau; enlarging the globally coupled credit
window cannot approach the target.

Next, the bank-run cap was connected to the existing receiver batch-row test
control.  This increases the number of L1 slots while holding logical bytes
constant, but it also increases Fabric command count.  All cases passed the
safe pytest runner:

| Rows per packet | Median | Effective receive BW |
| ---: | ---: | ---: |
| 20 | 5.721 ms | 55.682 GB/s |
| 8 | 8.753 ms | 36.391 GB/s |
| 4 | 14.942 ms | 21.319 GB/s |
| 2 | 27.899 ms | 11.418 GB/s |

Decision: preserve full 20-row/~14-KiB packets.  Extra globally shared slots
cannot offset packet issue overhead.  The next architectural experiment must
split the credit dependency itself: independent forward/backward and then
per-bank or per-destination staging/credit rings, while retaining coalesced
packets.  Bank-specific receiver fan-out is useful only when paired with those
independent credits; attribution shows the current drain engine is not the
standalone bottleneck.

### 2026-07-19: directional-credit experiment design

The next TDD experiment separates only credit fan-in; payload placement,
produced sequences, coalesced packets, and receiver drain order stay unchanged.
The current reader and writer both wait on one sender-local credit counter with
a threshold of `N-1` acknowledgements per group.  Consequently either ring
half can prevent both workers from reclaiming their next slots.

For an even 1D ring and an even, slot-aligned credit group:

1. Allocate independent reader/forward and writer/backward credit counters.
2. A destination's writer returns credit over its alternate backward range to
   the remote reader counter; a destination's reader returns credit over its
   primary forward range to the remote writer counter.
3. Both return ranges contain `N/2` destinations.  They deliberately both
   cover the opposite rank: within an even group, alternate-route balancing
   sends at least one batch to that rank from each payload worker.
4. Each payload worker may reclaim its slot group after `N/2`
   direction-specific credits, rather than after the shared `N-1` epoch.
5. The local consumed sequence remains shared because it is published only
   after the receiver has drained every remote source for the complete group.

The experiment is opt-in and initially restricted to the bank-owned 1D ring,
pipelined credits, an even group size, and a slot count divisible by the group.
This prevents a partial parity group or wraparound from acknowledging an
opposite-rank slot that the corresponding worker has not yet retired.

TDD order:

1. Pure reference proof that both directional counters receive exactly `N/2`
   credits and that every credited destination consumed that worker's payload
   within the group.
2. Parser/program-identity test for the opt-in control.
3. Repeated small 8x1 BF16 and FP8 exact-output tests.
4. Exact 64,640-row/rank FP8 A/B/A against shared group-4 credits.
5. Attribution only if the clean measurement clears the 3% retention gate.

### 2026-07-19: directional-credit result — rejected

- The pure schedule proof passed for 4- and 8-device rings with groups 2, 4,
  and 6.
- Implemented independent reader/writer credit counters, crossed return
  addresses, and a writer alternate-range credit header.  Small 8x1 BF16 and
  scaled-FP8 exact-output tests passed through the safe runner.  The initial
  small test correctly rejected its automatic 79-slot ring because group four
  did not divide it; pinning 12 slots exercised the intended bounded geometry.
- Exact 64,640-row/rank scaled-FP8 A/B/A, three samples per arm:

| Credit fan-in | Median | Effective receive BW |
| --- | ---: | ---: |
| shared A | 5.719 ms | 55.699 GB/s |
| directional | 5.792 ms | 54.993 GB/s |
| shared A | 5.739 ms | 55.510 GB/s |

Directional fan-in is about 1.1% slower than the mean shared control and fails
the 3% retention gate.  This shows that, at a four-batch group, waiting for the
other ring half is not the material head-of-line dependency.  The extra credit
header/state adds overhead without allowing payload production to advance.
Remove the experimental runtime path; retain only this result in the work log.
The next experiment must create multiple independently reusable payload slot
rings (per bank or bank group), rather than splitting acknowledgements for one
shared slot ring.

### 2026-07-19: per-bank receiver-fan-out design

The next prototype gives each of the four DRAM banks owned by a Fabric link a
dedicated receiver Tensix core.  It changes the amount of independent staging,
not the packet size:

- two existing sender cores remain, one per Fabric link;
- each sender targets four receiver cores selected from the packet's owned-bank
  slot, for eight receiver cores total;
- every receiver core gets its own `N * slots * 14 KiB` ordinary-L1 staging
  ring, produced epochs, and dual-RISC synchronization value;
- four consumed sequences and four sender credit counters make those rings
  independently reclaimable;
- sender source/local work remains coalesced into one physical-bank run;
- each receiver writes only its assigned output DRAM bank, so its NoC stream
  cannot serialize commands for the other three banks;
- control semaphores remain in L1-small.  Produced and dual-RISC global
  semaphore addresses can be shared because each receiver core has a distinct
  local instance; bank-specific sender credit/consumed addresses cannot.

The existing bank-major packet order is retained initially.  A sender keeps a
separate batch/slot/credit sequence for each bank, and targets the appropriate
receiver coordinate.  This isolates the fan-out mechanism from a simultaneous
schedule-order change.  Interleaving banks becomes a separate follow-up only
if fan-out is correct but one bank monopolizes the sender timeline.

Initialization must count all receiver cores.  The writer sends its start
handshake to every receiver core; the reader waits for `1 + banks_per_link`
local ready signals before advertising remote readiness.  Completion still
waits for exactly `N-1` remote workers after subtracting those local signals.

TDD order:

1. Reference mapping proof: the four receiver schedules are disjoint, cover
   exactly one link's pages, and each maps to one physical DRAM bank.
2. Reference slot proof: every bank advances and reclaims independently while
   preserving monotonic produced/consumed sequences.
3. Host resource and program-identity tests for 10 total cores and the extra
   L1-small controls.
4. One drain RISC, small 8x1 BF16 exact output.
5. Two drain RISCs, small 8x1 FP8 exact output, then five repeats of both.
6. Exact FP8 clean A/B/A against the single-receiver group-4 baseline.
7. Retain only above 3%; otherwise remove the runtime path and record why.

### 2026-07-19: per-bank receiver-fan-out result — rejected

- Implemented four independent receiver cores per link (one per owned DRAM
  bank), each with private staging slots, produced/consumed sequences, and
  sender credit state. The packet size and bank-major ordering remained fixed.
- The first small tests were accidentally run with the perf helper's default
  2D Fabric configuration. Tt-triage showed every sender blocked in the remote
  initialization barrier while the receiver cores waited for their first
  produced payload. Re-running the intended 8x1 ring topology resolved the
  hang; this was test configuration, not a ring-path correctness failure.
- On the ring, five repeated BF16 and five repeated scaled-FP8 exact-output
  tests passed, including the dual-RISC receiver drain. The release build also
  passed.
- Exact 64,640-row/rank scaled-FP8 A/B/A, three samples per arm:

| Receiver cores per link | Median | Effective receive BW |
| --- | ---: | ---: |
| one A | 5.711 ms | 55.774 GB/s |
| four, one per bank | 5.744 ms | 55.453 GB/s |
| one A | 5.709 ms | 55.797 GB/s |

Four receiver cores are about 0.6% slower than the mean single-receiver
control and fail the 3% retention gate. This confirms the receiver's DRAM
write fan-out is not limiting this case: attribution already showed the
receiver spends roughly 4.36 ms of its 5.49 ms critical path waiting for
payload production, while actual coalesced drain issue plus flush is about
1.13 ms. More destination write engines add synchronization and placement
overhead without increasing the arrival rate. The fan-out runtime control and
tests were removed; the single-receiver path remains the implementation.

The next useful experiment must raise sender-side payload arrival rate. Keep
20-row/~14-KiB packets and the 12-slot/group-4 receiver geometry, then isolate
the source-read-to-Fabric path with counters for DRAM read completion, Fabric
command issue, and downstream buffer availability. Candidate changes, in
order, are: overlap source DRAM reads with the prior Fabric packet, stage two
or more source packets per sender RISC, split source-bank reads across both
data-movement RISCs without fragmenting Fabric packets, and only then consider
a custom sender pipeline or additional Fabric links.

Final retained-path validation after removing the fan-out control:

- release build: `./build_metal.sh --release` passed with warnings as errors;
- exact 8x1 ring bank-owned schedule: BF16 and scaled FP8 passed twice each;
- generic 4x2 receiver path: BF16 variants passed, and scaled FP8 passed after
  fixing its test oracle to ignore the replicated TP mesh dimension;
- exact large scaled-FP8 run, five samples: median `5.715 ms`, minimum
  `5.706 ms`, p90 `5.730 ms`, effective receive bandwidth `55.743 GB/s`.
