# Native all-gather: contiguous receive plan

> Current qualification status (2026-07-21): the high-bandwidth bank-fanout
> receiver and cross-ERISC terminal-offload path is qualified after fixing its
> per-bank tail accounting. Full 65,536-row/device BF16 and scaled-FP8 output
> checks pass, repeated large perf fixtures complete without hangs, and both
> formats exceed 92 GB/s on Galaxy-compatible links. Older experiment tables
> below remain historical unless identified as current qualification.

## Summary

Improve small-row all-gather by changing the collective implementation, not the
MLA KV-cache layout.  The new fast path will retain the existing row-major,
one-row-per-page tensor contract, but transport a contiguous run of rows as one
logical transfer unit.  It will use a receiver Tensix core (or a small group of
receiver cores) to drain Fabric data from L1 to the persistent DRAM output.

This work is motivated by Sparse MLA KV gathers, where a BF16 row is 1152 B and
an FP8 row is about 704 B including DRAM-page alignment.  These are too small
to amortize the current all-gather's per-page read, Circular Buffer, and remote
scatter-write overhead.

The model must not need to pack several logical rows into a larger cache page.
The generic all-gather is responsible for recognizing contiguous physical and
logical ranges, coalescing them internally, and retaining the current generic
path as a fallback.

The current candidate has proved the connection-free receive protocol on a
4x2 Blackhole mesh and a supported 8x1 1D ring.  A receiver Tensix accepts
Fabric payloads in ordinary L1, drains them to the persistent output, and
returns consumption through a local semaphore.  The existing sender
reader/writer routes proxy the remote credits, so the receiver consumes no
additional Fabric client/channel.  Automatic selection now retains the
receiver for the measured cases where it wins and keeps the established path
for BF16 ring, small transfers, unsupported mappings, and insufficient
L1-small control capacity.  Receiver-confirmed SP=4 end-to-end translation is
complete on the available P150 4x2 system.  The remaining decision is
validation on actual QuietBox and Galaxy product topologies.

This document is both the design plan and the experiment ledger.  A result is
not actionable unless it records the exact code revision, command,
hardware/topology, raw samples, and whether the fast path or fallback ran.
The current post-T4 execution queue, goals, core/resource contract, commands,
and acceptance gates are maintained in
`docs/ALL_GATHER_TENSIX_RECEIVER_ITERATION_PLAN.md`.

## Current implementation and observed problem

The native multicast implementation currently allocates one Tensix worker per
Fabric link.  That worker:

1. issues one DRAM NOC read for each input page into its L1 Circular Buffer;
2. emits a Fabric write for each output chunk; and
3. lets `FabricWriter` pack at most four small chunks in one scatter packet.

The last point is important: Fabric payloads are already partially coalesced.
For a 1152 B row, four rows form a 4608 B scatter packet.  Thus, changing only
the nominal packet size cannot eliminate the principal small-row overhead.
The individual source reads, CB progress, output address generation, and
scatter-write destinations remain per-row.

The paired writer kernel also writes its local contribution one chunk at a
time.  The remote Fabric packet currently names final DRAM destinations.

The relevant implementation is:

- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_common.hpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp`

### Stage attribution update (2026-07-18)

The first two production-kernel experiments narrowed the bottleneck.  Both
were run on the same 4x2 Blackhole mesh, with 32,768 rows per device, one row
per page, persistent output, five to seven timed samples, and the safe pytest
runner.

| Variant | BF16, 1152 B/page | FP8 E4M3, 704 B/page | Result |
| --- | ---: | ---: | --- |
| Original one-row path | 3.538 ms / 32.01 GB/s | 3.390 ms / 20.42 GB/s | Baseline |
| Contiguous-address Fabric write fast path | 3.538 ms / 32.01 GB/s | 3.390 ms / 20.42 GB/s | Neutral for MLA |
| Split source reads across reader/writer RISCs and both NoCs | 3.648 ms / 31.05 GB/s | 3.429 ms / 20.18 GB/s | Regressed 3.1% / 1.2%; removed |

The contiguous-address writer optimization is correct and retains scatter as
fallback, but it does not trigger for the MLA one-row mapping: consecutive
logical pages are interleaved across DRAM banks and therefore do not form one
monotonic physical NOC-address run.  Ten focused row-major tests pass,
including BF16, FP8, grouped rows, tails, and persistent output.

The dual-NoC experiment establishes that adding sender-side read concurrency
does not improve this workload.  It does not prove that reads are free, but it
rejects source-read issue bandwidth as the first optimization target.  The
next experiment must therefore isolate destination command/address overhead
by terminating Fabric writes in Tensix L1 and draining to DRAM locally.

### Receiver prototype checkpoints (2026-07-18)

The first one-slot receiver prototype compiled in release mode and reached the
device, but the focused 4x2 MLA-shaped test hung.  The safe pytest runner
captured `generated/tt-triage/triage.csv` and reset the devices.  The important
stalls were:

| Kernel | Stalled operation | Meaning |
| --- | --- | --- |
| sender writer | wait for remote slot credit | the sender cannot reuse its receive slot |
| receiver, one link | wait for a source's `produced` sequence | not all payload streams reached that receiver |
| receiver, another link | Fabric credit send / empty-write-slot wait | the receiver could not make progress opening or using its return route |
| sender reader | final completion barrier | a consequence of the incomplete data/credit protocol |

NOC status and CB-inactive checks passed.  The failure is therefore consistent
with Fabric client/channel oversubscription, not corrupt destination NOC writes:
the prototype added a new direct Fabric connection from each receiver core
while the colocated sender reader/writer already owned the usable endpoint
routes.  Progress was asymmetric between links, which is the expected failure
mode if one extra connection acquires a channel and another waits indefinitely.

This is a design result, not a timeout problem.  The next prototype must not
open a Fabric connection from a receiver core.  Receiver cores will only accept
Fabric writes into L1 and drain L1 to DRAM.  They will signal consumption by a
local NOC semaphore increment to the associated sender core.  The existing
sender reader/writer Fabric routes will proxy the return credits to remote
devices.

The connection-free prototype implemented that design.  Its first run exposed
a second, independent completion bug: all receiver kernels and writer BRISCs
had finished, while every reader NCRISC remained at the final completion
barrier.  The same barrier value was used for initialization and completion.
Initialization legitimately waits for `N - 1` remote workers plus two local
receiver-reset acknowledgements; completion receives only the `N - 1` remote
worker signals.  Reusing the larger initialization threshold made completion
unreachable even though all payloads and credits had completed.

The fix keeps the initialization threshold at `N + 1` and uses `N - 1` for the
final completion barrier when receiver mode is active.  After that correction:

| Gate | Result |
| --- | --- |
| Release host build | PASS |
| Focused BF16 4x2 MLA-shaped gather | PASS |
| Focused BF16 repeated with program reuse | 20/20 PASS |
| Focused FP8 E4M3 equivalent | PASS |
| BF16 batch with a one-page final tail | PASS |
| Safe-runner hang recovery / tt-triage workflow | Exercised successfully on both protocol bugs |

These passes establish the one-slot protocol and epoch reset, not the final
performance case.  Partial gather, batch slice, explicit fallback selection,
the wider row-major matrix, and product topologies remain open.

### Evidence so far

These results are diagnostic measurements from an 8-chip 1D loudbox proxy with
a 500K-token BF16 cache, not a Galaxy product claim:

| Transport experiment | Latency | Effective BW | Interpretation |
| --- | ---: | ---: | --- |
| Current one 1152 B row/page, 14 KiB packet setting | 14.454 ms | 36.064 GB/s | Current baseline |
| Four contiguous rows represented as one 4608 B page | 7.781 ms | 66.989 GB/s | 1.86x faster with unchanged payload |
| Same grouped transfer on 1D ring proxy | 6.411 ms | 81.311 GB/s | Ring uses both directions effectively |

FP8's smaller payload also delivered much less latency improvement than its byte
reduction predicts.  Together, these results point to transfer-command and
pipeline overhead rather than pure Fabric payload bandwidth.  They do *not*
by themselves establish the exact split between source DRAM reads,
Fabric/ERISC processing, and destination DRAM writes.  The command-count and
cycle-attribution checkpoint below now supplies that split.

### Current measured checkpoint and active limiter (2026-07-18)

The receiver implementation has advanced beyond the early checkpoint above.
The following results are the current starting point for new work:

| Gate | Current result |
| --- | --- |
| Release build | PASS with `./build_metal.sh --release`; no `--dev` |
| Isolated direct BF16 / receiver BF16 | 3.540 / 3.063 ms, a 13.5% receiver win |
| Isolated direct FP8 / receiver FP8 | 3.373 / 1.753 ms, a 48.0% receiver win |
| Full Sparse MLA correctness | 39/39 PASS after persistent-output integration across BF16/scaled-FP8 and 2x4/4x2 (950.50 s on the final release-built topology-policy candidate) |
| Focused RM receiver matrix | 11/11 PASS, plus 20 repeated runs per dtype |
| Full local all-gather file | 66 PASS / 4 intentional opt-in skips in 66.37 s with realtime-profiler path assertions; the skips are two diagnostic bank-owned cases and two isolated perf cases already measured separately |
| Sparse MLA model perf on the available SP=4 x TP=2 proxy | Receiver-confirmed A/B/A: warm total gains 0.35-0.47% BF16 and 4.40-5.58% FP8; long gains 3.99-5.11% BF16 and 12.13-15.69% FP8 |
| Supported 8x1 1D ring proxy | FP8 receiver wins 16.6%; BF16 receiver loses 26.8% and automatic selection keeps direct/unicast |
| QuietBox and actual Galaxy Torus-Y/ring | Still required; the 8x1 P150 result is not a Galaxy claim |

The final local inventory check with `tt-smi -s` exposes eight `p150b` boards
and no QuietBox or Galaxy devices.  Product-topology validation therefore
cannot be inferred from this host and remains a separate hardware gate.

### Sparse MLA persistent-output and end-to-end checkpoint (2026-07-18)

The first model-level receiver comparison was invalid: MLA did not provide a
persistent output, so both nominal direct and automatic runs contained zero
receiver programs.  Integration now allocates one shared, replicated,
interleaved DRAM scratch tensor for the main KV-prefix gather.  The receiver
accepts the real ND-sharded DRAM cache only when its source and output page
geometry match exactly; the generic `TensorAccessor` remains responsible for
the source mapping.

`ttnn.all_gather(..., output_tensor=...)` returns a distinct Python wrapper for
the supplied allocation.  The model returns the cache-owned wrapper instead of
that temporary alias so normal downstream cleanup cannot deallocate the shared
buffer after the first forward.  The red cold test failed on its second gather
with `Input Tensor is not allocated`; after the correction, BF16 and FP8 each
completed 11/11 forwards and observed 11 receiver programs.  The complete
Sparse MLA suite then passed 39/39.

The receiver-confirmed SP=4 x TP=2 A/B/A result is:

| Case | BF16 total gain, DeepSeek / GLM | FP8 total gain, DeepSeek / GLM | BF16 CCL gain, DeepSeek / GLM | FP8 CCL gain, DeepSeek / GLM |
| --- | ---: | ---: | ---: | ---: |
| warm | 0.47% / 0.35% | 4.40% / 5.58% | 10.34% / 10.95% | 36.55% / 36.51% |
| long | 3.99% / 5.11% | 12.13% / 15.69% | 13.67% / 13.76% | 42.42% / 42.63% |

Automatic values are the mean of two A legs around a forced-direct B leg.
Every automatic warm/long run observed one receiver program, and every direct
run observed zero.  Direct and automatic device-program counts are identical:
58/60 for DeepSeek BF16/FP8 and 71/73 for GLM BF16/FP8.  CCL counts also remain
five for DeepSeek and four for GLM; the receiver is a kernel implementation
inside an existing collective program, not an additional model operation.  The
full raw table and reproduction command are in
`models/demos/deepseek_v3_d_p/tests/sparse_mla/sparse_mla_all_gather_perf.md`.

### Supported 8x1 ring and automatic-routing checkpoint (2026-07-18)

This checkpoint used an 8-chip P150 Blackhole system, an 8x1 mesh,
`FABRIC_1D_RING`, release firmware, row-major one-row pages, persistent DRAM
output, 2048 B of configured L1-small, 32,768 rows per device, and seven timed
samples.  Effective bandwidth counts the bytes received by one device.  The
direct control is the established ring/unicast implementation; the receiver
candidate is the multicast receiver implementation.

| Dtype | Direct median / min / p90 | Receiver median / min / p90 | Receiver delta | Automatic decision |
| --- | ---: | ---: | ---: | ---: |
| BF16, 1152 B/page | 4.363 / 4.346 / 4.369 ms | 5.531 / 5.520 / 5.536 ms | 26.8% slower | direct/unicast |
| FP8 E4M3, 704 B/page | 3.890 / 3.889 / 3.892 ms | 3.245 / 3.233 / 3.260 ms | 16.6% faster | receiver |

Raw samples, in milliseconds:

- BF16 direct: `4.363, 4.369, 4.366, 4.346, 4.357, 4.349, 4.363`
- BF16 receiver: `5.522, 5.520, 5.536, 5.533, 5.531, 5.534, 5.525`
- FP8 direct: `3.892, 3.889, 3.891, 3.890, 3.890, 3.891, 3.890`
- FP8 receiver: `3.255, 3.260, 3.245, 3.241, 3.260, 3.233, 3.236`

The resulting automatic ring policy selects the receiver only for FP8 when
the effective selected height is at least 32,768 pages.  It uses
`valid_gather_extent`, when present, rather than the allocated cache height.
An FP8 tensor allocated for 32,768 rows with an effective extent of 128 rows
therefore selected the direct partial-gather path and passed the profiler path
assertion at 0.046 ms.  This prevents a small prefix or batch slice from
inheriting a production-size policy decision from unused cache rows.

The small-transfer crossover also does not justify automatic selection.  At
128 FP8 rows per device, direct and forced receiver medians were 0.060 and
0.059 ms, while receiver p90 was worse (0.063 versus 0.061 ms).  The difference
is below the 3% retention gate.

The old forced direct/receiver experiment used private environment controls
that have since been removed. The supported reproducible command now exercises
the automatic production policy:

```bash
TT_METAL_DEVICE_PROFILER=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf and fp8' -q -s
```

Replace `fp8` with `bf16` for the BF16 case. Historical forced-path A/B
measurements remain useful evidence, but their private controls are no longer
part of the executable interface.

#### Torus-Y proxy control and automatic-policy boundary

The available P150 was also configured as a 4x2 `FABRIC_2D_TORUS_Y` proxy and
run with the same isolated harness at 128 BF16 rows/device.  Both forced
receiver and forced direct timed out on the first synchronization.  The safe
runner captured `generated/tt-triage/triage.csv`, reset all eight devices, and
reported a Fabric-progress deadlock.  In the receiver run, some readers never
completed the initialization barrier, some senders waited for Fabric write
slots or consumed credits, and all receivers waited for missing per-source
produced sequences.  The forced-direct control also stalled across multicast
reader/writer roles, so this P150 Torus-Y result is not evidence of a
receiver-specific defect and cannot substitute for a Galaxy run.

This control exposed a separate host-policy issue: per-axis topology reports
both an eight-device 1D ring and a Torus-Y axis as `Topology::Ring`.
Automatic selection now includes `FabricConfig` in the reflected operation
attributes/program-cache identity and enables the measured ring policy only
for exactly eight devices under `FABRIC_1D_RING`.  Other ring sizes and all
Torus configurations retain the established path until they have their own
correctness and A/B record; forced receiver remains available for bring-up.
After this guard, the release build passed, 4x2 linear auto selected the
receiver and passed, and 8x1 FP8 auto selected the receiver and passed at
3.249 ms for a one-sample path check.

#### L1-small capacity proof

Payload slots remain in ordinary L1.  Only the global barrier and receiver
control semaphores use L1-small.  The host computes the aligned structural
requirement before allocating the common barrier:

```text
1 global barrier + N device controls + 2 receiver controls
                 + 1 extra control when dual-RISC drain is selected
```

At 16-byte alignment this is 112 B for four-device BF16, 176 B for eight-device
BF16, and 192 B for eight-device dual-RISC FP8.  A focused four-device test with
only 64 B configured proves both branches: automatic mode falls back and
produces correct output, while forced receiver mode rejects before device
dispatch with the precise `control semaphores exceed` reason.  This is a proof
of configured structural capacity; transient runtime L1 allocation pressure
remains allocator-managed and is not claimed as host-proved free capacity.

The current production-sized attribution case uses 32,768 one-row pages per
device on a 4x2 Blackhole mesh and seven timed samples.  Command counts are
stable across all seven samples:

| Dtype | Source reads per directional worker | Fabric payloads per directional worker | Rows per Fabric payload | Remote DRAM writes on critical receiver RISC |
| --- | ---: | ---: | ---: | ---: |
| BF16, 1152 B/page | 16,384 | 1,366 | up to 12 | 49,152 |
| FP8 E4M3, 704 B/page | 16,384 | 820 | up to 20 | 23,592 with the remainder on the second drain RISC |

This proves that Fabric payload coalescing is already working: the sender does
not emit one Fabric packet per row.  The remaining high-frequency commands are
the source DRAM read, local-output write, and receiver DRAM write issued once
per tensor page.

Cycle attribution further localizes the work:

| Dtype | Source-read issue | Writer Fabric issue | Receiver payload wait | Receiver DRAM-write issue | Critical receiver total |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.842 ms | 1.107 ms | 0.384 ms | 2.632 ms | 3.067 ms |
| FP8 E4M3 | 0.840 ms | 0.510 ms | 1.027 ms | 0.810 ms | 1.844 ms |

Two conclusions set the next iteration order:

1. Source-read issue time is almost byte-independent: FP8 transfers 39% fewer
   bytes but still spends about 0.84 ms issuing 16,384 source reads.  The next
   source experiment must reduce commands, not merely add a second NOC.
2. BF16 is receiver-write limited, while FP8 is producer/Fabric-wait limited
   after its drain was split across two RISCs.  A single optimization is
   unlikely to be best for both dtypes.

An A/B/A experiment removed the `noc.async_writes_flushed()` immediately before
each receiver-L1 Fabric payload while retaining the end-of-batch flushes.  It
passed BF16 and FP8 correctness but was neutral: BF16 was 3.067 versus 3.066 ms
and FP8 was 1.868 versus 1.865 ms.  The experiment was removed.  Flush ordering
is therefore not the active target.

#### Physical-run histogram and bank-major experiments

Opt-in device-side address attribution now compares raw NOC addresses without
changing the production data path.  On Blackhole, logical row pages rotate
across eight DRAM banks:

| Mapping observation, critical core | BF16 | FP8 E4M3 |
| --- | ---: | ---: |
| Source logically adjacent pairs | 16,383 | 16,383 |
| Source adjacent pairs on the same bank | 0 | 0 |
| Source pages contiguous with the page eight rows earlier | 16,376 | 16,376 |
| Receiver adjacent pairs on the same bank | 0 | 0 |
| Receiver pages with a same-batch contiguous bank predecessor | 16,380 | 15,336 on the critical FP8 drain RISC |

This proves that coalescing is physically legal only after converting each
logical batch to bank-major order.  Three TDD variants implemented that
permutation consistently across source reads, local output, receiver-L1
payload, and remote output:

| Variant | BF16 median | FP8 median | Decision |
| --- | ---: | ---: | --- |
| Untouched A/A envelope | 3.060–3.069 ms | 1.864–1.868 ms | Baseline |
| Source and destination coalesced | 3.114 ms | best 1.846 ms | BF16 regressed; FP8 gain about 1.1%, below 3% gate |
| Source coalesced, destination per-page | Not pursued after BF16 rejection | 1.870 ms | Neutral |
| Source per-page, destination coalesced | Not pursued after BF16 rejection | 1.879 ms | Regressed |

The combined FP8 prototype reduced source commands from 16,384 to 6,556 and
critical receiver writes from roughly 24K to 9,444.  Source issue fell from
0.840 to 0.356 ms and receiver drain issue from 0.810 to 0.442 ms.  However,
the required per-payload permutation and changed sender pipeline increased
writer Fabric issue from 0.510 to 0.697 ms and receiver payload wait from 1.027
to 1.347 ms.  Skipping the pre-payload flush specifically for this variant was
also neutral.

All variants passed their focused correctness gates, but none met the retained
change threshold, so their data-path code and test controls were removed.  The
address attribution remains because it is useful general observability.

The next design must avoid permuting every payload on one sender core.  The
preferred direction is static bank ownership: a small set of source workers
reads already-contiguous bank runs into bank-major L1, sends through a Fabric
mux, and targets matching receiver workers that own disjoint output banks.
Core count must start at two, not eight, and scale only with measured gain.

## Goals

The goals below are ordered.  A later performance goal cannot waive an earlier
correctness or protocol goal.

### Functional goals

1. Preserve the public `ttnn.all_gather` contract, including arbitrary generic
   page mappings and the current direct scatter implementation.
2. Add a fast path when a batch of input rows has a host-provable page mapping
   that a receiver can drain safely.  Physical contiguity is required only for
   a single contiguous NOC command; receiver-side bank-aware scatter remains a
   valid optimized path for interleaved pages.
3. Preserve support for persistent/preallocated outputs, L1-small semaphore
   allocation, `batch_slice_idx`, and `valid_gather_extent`.
4. Keep MLA KV caches row-major and one logical row per physical tensor page.
   No model-side re-layout or cache padding shall be a prerequisite for the
   fast path.
5. Produce bitwise-equivalent BF16 output and the expected FP8 equivalence
   tolerance for full, sliced, and partial gathers.
6. Support the deployed product topologies, including Galaxy
   `FABRIC_2D_TORUS_Y` (SP ring, TP linear).  Ring is a supported deployment
   mode, not merely a benchmark configuration.

### Performance goals

All targets must be assessed against the current native multicast AG using the
same persistent output, tensor allocation, warmup, sample count, routing, and
Fabric configuration.

1. Explain at least 90% of baseline elapsed time with kernel/NOC/Fabric timing
   or a bounded residual before choosing a final design.
2. For contiguous MLA-shaped rows, recover at least 75% of the diagnostic
   four-row grouping gain without changing tensor page size.  On the stated
   proxy this means a first target of at most 9.5 ms from the 14.454 ms
   baseline; it is a directional engineering gate, not a product KPI.
3. Improve FP8 latency materially relative to its one-row native baseline;
   reducing bytes alone is not sufficient evidence of a good implementation.
4. Do not regress generic/small all-gather cases by more than 3% versus the
   current path.  Non-contiguous cases should select the fallback directly.
5. Compare linear and ring/torus routing on actual supported hardware.  Choose
   the topology by end-to-end throughput and compatibility, not proxy results.

### Engineering goals

1. No hangs, watcher/LLK assertions, or firmware-size regressions.  Do not use
   `--dev` for these runs because it changes firmware code size.
2. Every new fast-path predicate is host validated before dispatch: transport
   payload limits, L1 allocation, available CCL cores, and supported mapping.
3. All receive buffers have explicit producer/consumer credits and monotonically
   increasing sequence numbers.  A sender must never reuse an L1 slot before
   the owning receiver acknowledges consumption.
4. The implementation remains debuggable with `scripts/run_safe_pytest.sh` and
   tt-triage.  A hang is a test failure and requires a minimal reproducer before
   expanding the experiment.

### Current milestone goals

| ID | Goal | Measurable completion criterion |
| --- | --- | --- |
| G0 | Preserve the baseline | Record BF16 and FP8 isolated AG results from the same commit, hardware, topology, warmup, and sample configuration used by every candidate. |
| G1 | Prove the protocol | The connection-free one-slot receiver passes the focused 4x2 correctness case 20 consecutive times with no hang, timeout, stale semaphore, or device reset. |
| G2 | Prove coverage | BF16 and FP8 pass full batches, one-page tails, partial gather, batch slice, persistent output, and explicit fallback cases. |
| G3 | Locate the limiter | Measure L1-sink-only, one-NoC drain, and direct-to-DRAM baseline separately, so Fabric ingress and receiver DRAM drain are not conflated. |
| G4 | Win in isolation | Receiver AG improves median isolated latency by at least 5% with no p90 regression; otherwise it remains experimental and the reason is documented. |
| G5 | Scale with bytes | FP8 shows a meaningful latency reduction versus BF16 after fixed overhead is reported; if not, identify the fixed-cost stage before adding cores. |
| G6 | Generalize safely | Unsupported mappings and resource plans select the existing scatter path before device dispatch and regress it by no more than 3%. |
| G7 | Validate the product path | Sparse MLA correctness passes and end-to-end perf improves on the target 4x2 path; then validate QuietBox/loudbox and Galaxy Torus-Y/ring configurations. |

The 5% gate in G4 is an iteration gate, not the final target.  It prevents
adding buffering and fan-out to a design that has not yet demonstrated a real
gain.  The final target remains recovering at least 75% of the diagnostic
four-row grouping gain.

### Status and next decision gates

| Goal | Status on 2026-07-18 | Next evidence required |
| --- | --- | --- |
| G0 — reproducible baseline | Met on 4x2 Blackhole | Preserve direct/candidate raw samples and interleave A/B/A after every retained kernel change. |
| G1 — protocol stability | Met for focused 4x2 | Repeat the 20/20 record after any credit, slot, or core-ownership change. |
| G2 — correctness coverage | Met for current predicate | Keep the 11/11 RM matrix and 39/39 Sparse MLA suite green; expand only with the predicate. |
| G3 — locate limiter | Met for current path | Command counts and stage cycles identify BF16 receive-write issue and FP8 producer wait as the critical stages. |
| G4 — isolated win | Met | Current receiver improves BF16 13.5% and FP8 48.0% at 32K rows/device. |
| G5 — byte scaling | Met diagnostically | Effective bandwidth is 37.0 GB/s BF16 and 39.5 GB/s FP8 after dtype-aware placement; fixed per-page source issue explains why byte reduction alone did not scale initially. |
| G6 — generic safety | Met on reference 4x2 and supported 8x1 ring | Host plan validates mapping, buffers, payload, cores, configured L1-small bounds, and structural policy identity; the final full local file is 66 PASS / 4 intentional opt-in skips with profiler path assertions. Product topology remains G7. |
| G7 — product path | Met on available P150; product sign-off in progress | Sparse MLA correctness is 39/39, receiver-confirmed SP=4 A/B/A records warm and long wins, and supported 8x1 ring A/B is recorded; validate QuietBox and actual Galaxy Torus-Y/ring. |

The dependency status in this table deliberately prevents an attractive
end-to-end number from bypassing the isolated correctness and attribution
gates.

## Proposed architecture

### Data path

```text
source DRAM banks
   │  pipelined, bank-aware NOC reads into one logical transport batch
   ▼
source Tensix L1 staging / send core
   │  Fabric packet: destination = mirrored receiver Tensix L1 slot
   ▼
ERISC Fabric route
   ▼
destination Tensix L1 receive slot
   │  receiver observes source-indexed produced sequence
   ▼
destination Tensix write core
   │  bank-aware posted NOC write(s), coalesced only where addresses prove contiguous
   ▼
persistent output DRAM
```

Control travels on a separate logical path:

```text
receiver drains L1 slot and flushes its local NOC writes
   │ local NOC atomic increment: consumed[sequence]
   ▼
associated sender core
   │ existing Fabric route proxies one credit to each remote source core
   ▼
remote source credit[sequence]
   │
   └── source may reuse its L1/CB slot only after all destinations acknowledge
```

The receiver owns no Fabric connection in this design.  This is the critical
difference from the first hanging prototype.

Fabric packets can name a Tensix L1 NOC address rather than a DRAM address.
Writing a packet does not invoke a Tensix callback, therefore the destination
core runs a persistent dataflow kernel which polls/waits on a semaphore and
drains assigned L1 slots to DRAM.

### Core allocation

For each Fabric link allocate from the CCL sub-grid:

- one source reader/sender core;
- one or more mirrored destination receiver/writer cores; and
- if a later design adds multiple Fabric-producing sender cores on one link,
  one Fabric mux core where the endpoint requires it.

Allocation must be deterministic and mirrored across all devices, must avoid
model/compute cores, and must validate that every requested core is available.
The initial prototype should use one sender and one receiver per link.  The
second prototype may stripe batch sequence numbers over two receiver writers
per link only after the one-receiver version is correct and profiled.

Each receiver owns disjoint output ranges and a ring of L1 slots.  A starting
configuration is three slots (to match current CB pipelining), sized to one
transport batch, plus L1-small semaphores for `produced`, `consumed`, and
completion.  Exact slot count and size must be derived from available L1 and
measured; they must not be hard-coded for a single model shape.

The receiver must not be counted as a Fabric-producing worker.  Its only
outgoing operation is a local NOC atomic increment to its associated sender
core.  This keeps the number of Fabric clients identical to the existing AG.

Core selection must be DRAM-bank aware, not just geometrically convenient.
At program creation time, enumerate the physical NOC destinations for the
pages in one representative batch, then score candidate CCL-grid cores by:

1. whether NOC0 and NOC1 can split the active bank set without duplicating an
   output page;
2. balanced bytes and command counts per NOC/RISC;
3. non-overlap with Fabric-router and model compute cores;
4. sufficient L1 for all slots plus control state; and
5. a deterministic mirrored choice on every device.

The first receiver prototype uses one core and one NOC so correctness and
credit behavior are unambiguous.  The next variant statically assigns disjoint
bank/page lists to NOC0 and NOC1.  Only after that wins may a second receiver
core be introduced.  A second core receives disjoint transport batches or
disjoint output ranges; two cores must never write the same DRAM cache line.

### Fast-path eligibility

The host selects the direct-contiguous path only if all of the following hold:

1. The source batch is physically contiguous in DRAM.
2. Its final output batch on every destination is physically contiguous in
   DRAM.
3. The batch size is within the Fabric/NOC payload and L1 staging limits.
4. Each batch is assigned to exactly one receiver slot and one final output
   range.
5. The device topology and available sub-grid satisfy the requested core plan.

The output iterator already knows page ID, byte offset, stripe length, and
stripe jump.  Extend it with a bounded look-ahead which proves contiguity for a
batch.  If it cannot prove it, dispatch the existing scatter path.  Do not
attempt to infer contiguity only from logical tensor shape; sharding and
partial-gather mappings can break it.

For MLA prefix gathers, the normal per-device contribution stripe is expected
not to qualify for a single physical contiguous write because row pages are
interleaved across banks.  The receiver path has a separate eligibility proof:
the host must be able to generate a bounded page list for each batch, assign
every destination page to exactly one receiver/NOC, and fit payload plus
control state in L1.  `batch_slice_idx` and `valid_gather_extent` must be
included in both proofs and in the test matrix.

## Iteration plan

Each phase is independently tested and benchmarked.  Do not combine phases or
land model changes to hide an AG regression.

### Phase 0 — establish attribution and reproducible baseline (completed on 4x2)

**Question:** Which portion of current time is source read, Fabric transfer,
remote output write, and synchronization?

1. Keep the existing isolated AG perf test and add profiler markers around:
   source DRAM reads, CB wait/push/pop, Fabric send/credit wait, receiver/local
   output writes, and completion barriers.
2. Run rows of approximately 704 B (FP8) and 1152 B (BF16), plus 2, 4, and 8
   contiguous rows per transfer as an instrumentation-only reference.
3. Sweep packet size only within valid hardware limits.  Reject oversized
   payloads before program compilation/dispatch.
4. Collect timings for linear and ring/torus routing separately.

**Exit criteria:** The timing report accounts for at least 90% of wall time or
shows a clear measurement limitation.  We know whether the first optimization
should target source reads, destination writes, or both.

### Phase 1 — source-read attribution (completed, rejected as default)

**Question:** Can one reader issue fewer/larger contiguous DRAM reads while
keeping the current direct-to-DRAM Fabric destination?

1. A dual-NoC prototype split each source batch across the reader and writer
   data-movement RISCs while retaining output addressing and Fabric scatter.
2. The full focused functional matrix passed.
3. BF16 regressed 3.1% and FP8 regressed 1.2%, so the prototype was removed
   rather than being enabled for MLA.

**Why first:** This isolates source DRAM command overhead from the receive
design and retains the current proven completion protocol.

**Decision:** The exit criterion was not met.  Retain the measurements as
stage-attribution evidence and do not add sender-side dual-NoC synchronization
to the production path.

### Phase 2 — contiguous direct remote-DRAM packet (implemented, neutral for MLA)

**Question:** Is the remote scatter command/header path the main remaining
cost when final output rows are consecutive?

1. `FabricWriter` now verifies queued physical NOC addresses and emits one
   contiguous Fabric unicast write when every next address equals the previous
   address plus page size.  Otherwise it emits the original scatter write.
2. Retain current scatter writes for all other mappings.
3. Benchmark this independently of Phase 1, then combine them.

**Decision:** Correct and neutral for MLA one-row pages because their physical
addresses are interleaved.  Keep it as a generic fast path subject to broader
regression testing; proceed to receiver staging for MLA.

### Phase 3 — one connection-free receiver Tensix writer per link (completed for 4x2)

**Question:** Can destination-side Tensix issue DRAM writes more efficiently
than the direct Fabric remote-DRAM path?

1. Keep the prototype behind a narrow, host-proved predicate in the multicast
   factory.  Its first test shape is the isolated 4x2 MLA CCL case; model
   integration remains unchanged until the isolated path is stable.
2. Allocate a dedicated mirrored receiver core and one source-indexed L1 slot
   per remote sender.  Source-indexed slots prevent simultaneous senders from
   targeting the same L1 bytes.  Add a second slot/source only after the
   one-slot protocol is stable and its lack of overlap is measured.
3. Send each qualifying batch to its receiver L1 slot with a normal Fabric
   unicast payload whose destination NOC address is the Tensix L1 slot, not
   DRAM.  Send the produced semaphore increment only after the payload command
   on the same ordered Fabric stream.
4. Precompute the receiver's output page list and split it into disjoint
   per-NOC lists.  The first implementation drains with one NOC; a later A/B
   uses both RISCs/NOCs.  Each RISC issues local NOC writes from contiguous L1
   offsets to the exact DRAM page addresses from its list.
5. Keep payload slots in ordinary L1 and allocate `produced`, `consumed`, and
   completion semaphores in L1-small.  Persistent output is mandatory in both
   baseline and candidate measurements.
6. Use a producer/consumer credit protocol without receiver Fabric clients:
   - the sender writer resets its local `credit` and `consumed` semaphores;
   - the receiver resets every source-indexed `produced` semaphore and signals
     local initialization readiness;
   - a sender writes its payload to the remote L1 slot and then increments the
     matching `produced[source]` on the same ordered Fabric stream;
   - the receiver waits for the expected sequence from every contributing
     source, writes the exact interleaved DRAM page list, flushes local NOC
     writes, and locally increments `consumed` on its associated sender core;
   - the existing sender reader/writer RISCs wait for `consumed`, then use their
     already-open, disjoint Fabric routes to return credits on behalf of the
     local receiver;
   - the source writer reuses its slot only after its local `credit` count
     proves that all remote destinations consumed that sequence.
7. Implement and measure three sub-stages independently:
   - L1 sink: Fabric payload and semaphore arrive, receiver validates a sample
     or checksum but performs no DRAM drain;
   - one-NoC drain: receiver writes the real persistent output;
   - two-NoC drain: reader/writer RISCs drain disjoint bank/page lists.
8. Start with one batch in flight; validate ordering, tails, and partial
   gathers before increasing slot count.

**Exit criteria:** G1 through G4 pass.  If staging adds more cost than it
removes, do not pursue multi-writer fan-out.

#### Phase 3A — repair the control protocol (completed)

This iteration is implemented and has passed the focused stability gate.

1. Remove Fabric connection, routing arguments, packet-header allocation, and
   connection close from the receiver kernel.
2. Reserve L1-small control semaphores as:
   `credit`, `consumed`, and `produced[num_devices]`.  Keep payload slots in
   ordinary L1.
3. Have the receiver locally increment `consumed` only after its posted DRAM
   writes are flushed and the source L1 bytes are safe to reuse.
4. Reuse the sender reader/writer semaphore route for remote credit return.
   Each RISC returns credits only over its existing destination subset; the
   subsets must be disjoint and cover every remote destination exactly once.
5. Keep one batch in flight and monotonically increasing sequence values.  Do
   not add slot rings, dual-NoC drain, or another receiver until this version
   passes G1.
6. Add compile-time/runtime assertions for expected credit count, route
   partition, L1 bounds, and semaphore count.

The protocol invariants are:

- one writer owns initialization of each local semaphore;
- payload precedes `produced` on the same ordered stream;
- `consumed(n)` implies every remote source slot for batch `n` has been read;
- exactly one credit per destination reaches each source for batch `n`;
- a slot for batch `n + 1` is not written before all credits for batch `n`;
- completion is sent only after the last data and credit command on each route.

Implementation checkpoint:

- receiver control layout is `credit`, `consumed`, then one
  `produced[source]` semaphore per device;
- payload slots use ordinary L1 and control semaphores use global L1-small;
- the receiver has no Fabric routing runtime arguments, packet-header buffer,
  connection open, or connection close;
- reader and writer proxy disjoint credit subsets through their existing
  routes and wait for the aggregate local credit before slot reuse;
- receiver completion uses the remote-worker threshold, excluding the two
  initialization-only local reset acknowledgements.

Do not alter these invariants while measuring the one-slot candidate.  Slot
rings or windowed credits are separate experiments and must get their own
stability run.

#### Phase 3B — correctness ladder

Run the smallest case first and advance only when the previous rung is stable:

1. one focused BF16 4x2 case, one iteration;
2. the same case 20 consecutive iterations;
3. FP8 equivalent;
4. a batch tail and a one-page transfer;
5. persistent-output full gather;
6. `valid_gather_extent` and `batch_slice_idx` independently and together;
7. explicit ineligible mapping to prove fallback selection;
8. the focused row-major correctness matrix.

All eight rungs now pass for the current predicate.  The focused RM matrix is
11/11, repeated stability is 20/20 per dtype, and the complete Sparse MLA suite
is 39/39 across BF16/scaled-FP8 and 2x4/4x2.  Repeat this ladder after any
change to page ownership, batching, slots, credit ordering, or fallback
eligibility.

On any hang, preserve the safe runner's tt-triage output and record, per stuck
core: kernel, source line, semaphore address/value, expected value, Fabric
connection state, and last completed sequence.  Fix the minimal case before
moving up the ladder.

#### Phase 3C — stage-isolation measurements

Measure these variants with identical dispatch and synchronization:

1. **L1 sink:** receive payload and validate arrival, but omit output DRAM
   drain from the timed steady state where correctness permits.
2. **One-NoC drain:** current receiver writes the real persistent output.
3. **Drain-only calibration:** prefill receiver L1 locally and measure only the
   L1-to-interleaved-DRAM schedule.
4. **Direct remote DRAM:** existing production fallback baseline.

Interpretation:

- L1 sink no faster than direct DRAM: Fabric packet/command issue is limiting;
- fast L1 sink, slow drain-only: destination address generation/NOC/DRAM is
  limiting;
- both micro-stages fast but combined slow: credit serialization or loss of
  overlap is limiting;
- BF16 improves but FP8 does not: fixed command/synchronization overhead still
  dominates the smaller payload.

The stage modes must not change host dispatch count, tensor allocation,
persistent-output allocation, route setup, warmups, or sample collection.  An
L1-sink measurement may omit the final output drain only inside the device
kernel; otherwise it would measure a different host program.  Drain-only must
use the same receiver page list and L1 offsets as the combined path.

For each dtype, fit the minimal model

```text
T(total_bytes, batches) = T_fixed + batches * T_batch + total_bytes / BW_payload
```

using at least three transfer sizes and two batch sizes.  The fit is diagnostic,
not a performance claim: report the residual and do not extrapolate beyond the
measured range.  This separates the reason FP8 does not scale from its reduced
payload into fixed dispatch/completion cost, per-batch credit/command cost, and
byte-dependent transport/drain cost.

#### Phase 3D — controlled tuning

After G1 through G3 pass, change exactly one parameter per measurement:

1. batch rows/bytes: 1, 2, 4, 8, bounded by packet and L1 limits;
2. slot count: 1, 2, 3;
3. credit granularity: per batch, then per slot window if correctness allows;
4. receiver page ordering grouped by DRAM bank/NOC destination;
5. source/receiver core placement scored for active DRAM banks.

Retain a change only if the median improves by at least 2%, p90 does not
regress, and the correctness ladder remains green.  Re-run the untouched
baseline between variants to detect thermal/system drift.

The first tuning order is intentionally fixed:

1. coalesce consecutive logical rows into a larger payload while preserving
   the one-row tensor page contract;
2. pipeline two L1 slots to overlap Fabric ingress with DRAM drain;
3. increase to three slots only if two slots still show receiver idle time;
4. reorder receiver writes by DRAM bank/NOC without changing ownership;
5. use both receiver RISCs/NoCs on disjoint page lists;
6. add a second receiver core only if one core is demonstrably drain-limited.

This order answers where the limit is before spending CCL cores.  In
particular, receiver fan-out cannot repair sender packet issue overhead, and
larger payloads cannot repair a receiver DRAM schedule that is already the
limiter.

### Phase 4 — reduce per-page DRAM command cost (completed, rejected for production)

**Question:** Can native AG reduce source and destination DRAM commands while
preserving the public one-row-per-page tensor layout?

1. Add a diagnostic address-run histogram for source and destination pages:
   consecutive physical address, same DRAM bank with a constant stride, and
   bank changes.  Do not assume logical row adjacency implies physical
   contiguity.
2. Prototype source command coalescing only where the tensor accessor proves a
   legal larger read.  If interleaving prevents this, test a bank-owned reader
   schedule that reorders commands into longer same-bank runs and restores
   logical payload order in L1.
3. Prototype receiver bank-owned draining.  Assign each destination page to
   exactly one RISC/core, group writes per DRAM bank/NOC, and compare the exact
   page list against the current order.  The existing FP8 dual-RISC split is
   the reference; BF16 must independently justify more receiver parallelism.
4. Add a second receiver core only after drain-only attribution proves the
   first core is still issue-limited and the extra core has a conflict-free
   bank/NOC allocation.
5. Keep Fabric batching unchanged initially.  It already emits one contiguous
   receiver-L1 payload for up to 12 BF16 or 20 FP8 rows; changing Fabric packet
   size at the same time would destroy attribution.

**Exit criteria:** A retained variant reduces the relevant per-page command
count or issue cycles, improves isolated median by at least 3%, does not regress
p90, passes A/B/A drift checks, and keeps the full correctness ladder green.
Stop when command count cannot legally be reduced for the interleaved mapping,
or when the critical stage moves to the Fabric link.

**Decision:** The phase reached the second stop condition.  Tail-aware bank
runs reduce source/local/Fabric commands to 1,368 BF16 and 820 FP8 per
directional worker.  They improve the static path by 14.4% BF16 and 8.2% FP8
over the exact-divisor control.  Receiver run coalescing remains slower than
source/local coalescing.  Most importantly, the best static path is 4.1%
slower BF16 and 1.7% slower FP8 than the accepted automatic receiver in clean
automatic/static/automatic measurements; control drift is only 0.16%.
Static bank ownership therefore remains diagnostic-only.  The receiver is
waiting for payload production, so additional receiver cores or receiver
write scheduling are not justified by current attribution.

### Phase 5 — product integration

1. Move the winning, bounded fast path into native all-gather dispatch.
2. Preserve direct scatter AG as default fallback for arbitrary mappings.
3. Enable it for MLA only through ordinary AG eligibility; no MLA-specific
   physical packing or distinct user-visible API.
4. Run functional Sparse MLA, Sparse MLA perf, CCL perf, and broader AG tests.
5. Validate relevant QuietBox/loudbox and Galaxy configurations.  On Galaxy,
   run both supported linear and `FABRIC_2D_TORUS_Y` configuration(s).

## Test-driven development plan

### Iteration discipline

Every experiment follows the same loop:

1. state one hypothesis and the metric expected to move;
2. make one architectural or tuning change in a standalone commit;
3. build with `./build_metal.sh --release`;
4. run the focused correctness case through `scripts/run_safe_pytest.sh`;
5. run the relevant correctness ladder;
6. collect interleaved A/B/A performance samples;
7. keep, revise, or revert based on the written gate;
8. update the results table with commit, command, hardware, and raw artifact.

Do not mix protocol repair, buffer-depth changes, page batching, core placement,
and dual-NoC fan-out in one experiment.  A standalone commit may be retained as
a diagnostic even when rejected, but rejected code must not remain enabled in
the candidate fast path.

### Unit correctness matrix

Add/retain focused tests under
`tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py`:

| Dimension | Required cases |
| --- | --- |
| Data format | BF16, FP8 E4M3 |
| Row size | MLA-like 1152 B BF16 and aligned FP8 row sizes; ordinary tile pages |
| Batch size | 1, 2, 4, tail of 1 after a full batch |
| Mapping | contiguous, non-contiguous fallback, sharded where supported |
| Output | freshly allocated and persistent/preallocated |
| Gather scope | full gather, `batch_slice_idx`, `valid_gather_extent` |
| Topology | 1D linear, ring, 2D linear, 2D Torus-Y where hardware is available |
| Core plan | one receiver/link; two receivers/link once Phase 4 starts |

For every fast-path test, assert both the numerical result and the selected
program path.  Test the fallback explicitly so an unsupported mapping never
silently receives an incorrect contiguous write.

Add non-public test observability before G2/G4 sign-off.  It must report one of
`direct_scatter`, `direct_contiguous`, or `receiver_l1`, together with the
eligibility rejection reason when fallback is selected.  The isolated harness
also needs an explicit `auto` / `force_direct` / `force_receiver` selection so
A/B runs do not depend on temporarily editing a predicate.  Forced receiver
mode must still validate all safety constraints and reject an ineligible case;
it is not allowed to bypass resource, mapping, packet-size, or topology checks.
L1-sink and drain-only remain test builds/modes and must not become public
`ttnn.all_gather` API options.

### Isolated performance matrix

Use the AG CCL perf test as the source of truth; retain the sparse-MLA CCL perf
test only as a workload-shaped cross-check.  Every result must report:

- topology and routing mode;
- tensor dtype, logical row bytes, physical page bytes, and total bytes;
- source batch rows/bytes and destination write rows/bytes;
- packet size, outstanding slots, links, sender and receiver core counts;
- persistent-output status and semaphore placement;
- median, minimum, p90, and effective payload bandwidth;
- warmup/sample count and git commit IDs.

Benchmark order:

1. current native AG baseline;
2. Phase 1 only;
3. Phase 2 only;
4. Phase 1 + Phase 2;
5. receiver path with one slot, then three;
6. dual-NoC / multi-receiver variants only if prior stage wins;
7. Sparse MLA end-to-end validation for BF16 and FP8.

For receiver experiments, also report three derived rates so a plateau has an
actionable owner: source bytes read per second, Fabric payload bytes delivered
to L1 per second, and destination bytes drained to DRAM per second.  If the L1
sink is not faster than the existing direct-DRAM path, stop: the Fabric/packet
front end is limiting and receiver fan-out cannot help.  If the L1 sink is
faster but one-NoC drain loses the gain, optimize bank scheduling/NoC writes.
If two-NoC drain is still below the L1 sink, then and only then add a second
receiver core.

Do not compare a composite/op-level timing to an isolated AG timing.  Record
AG elapsed time separately from model end-to-end time and state which is shown.

### Commands and failure handling

Run tests through the safe runner, without `--dev`:

```bash
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py -q -s
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -q -s
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py -q -s
```

Use the isolated benchmark's explicit environment gates rather than enabling
large perf sweeps by default.  Build candidate C++ changes with:

```bash
./build_metal.sh --release
```

When the safe runner reports a hang, preserve its tt-triage output, reduce to
the smallest single-AG reproduction, and fix the credit/termination protocol
before resuming performance sweeps.  Do not mask an issue by increasing timeouts
or changing firmware mode.

For the immediate focused case, use the existing MLA-shaped selector and do
not add `--dev`:

```bash
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'row_major_2k_pages and sp_axis_4x_mla_kv_bf16 and not 4rows' -q -s
```

The focused protocol ladder commands are:

```bash
# Program-reuse / semaphore-epoch stability.
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'row_major_2k_pages and sp_axis_4x_mla_kv_bf16 and not 4rows' \
  --count=20 -q -s

# FP8 transport geometry.
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'row_major_2k_pages and sp_axis_4x_mla_kv_scaled_fp8 and not 8rows' \
  -q -s

# Final batch containing exactly one page.
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'row_major_2k_pages and sp_axis_4x_mla_kv_bf16_one_page_tail' \
  -q -s

# Focused row-major matrix, including explicit fallback cases.
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'test_all_gather_fabric_2d_row_major_2k_pages' -q -s
```

Use the realtime device profiler for isolated measurements. The test fixes the
8x1 ring geometry, 65,536 rows per device, three samples, and both dtypes:

```bash
TT_METAL_DEVICE_PROFILER=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s
```

Copy the exact performance command and revision into the result row; a timing
without them is not a reusable measurement.

### Result record template

| Field | Value |
| --- | --- |
| Candidate commit / baseline commit | |
| Hypothesis | |
| Hardware and mesh | |
| Topology / routing | |
| Dtype / logical and physical row bytes | |
| Shape / gather dimension / partial range | |
| Persistent output / semaphore memory | yes / L1-small |
| Batch bytes / slots / sender and receiver cores | |
| Warmups / measured samples | |
| Median / min / p90 | |
| Effective payload bandwidth | |
| Correctness repetitions | |
| Safe-runner and tt-triage artifact | |
| Decision and reason | keep / revise / revert |

## Risks and decisions

| Risk | Mitigation / decision gate |
| --- | --- |
| Receiver staging costs more than remote DRAM scatter writes | Complete Phase 2 before committing to Phase 3; retain direct remote write when faster. |
| Receiver L1 overwrite or deadlock | Sequence-numbered credits, bounded slots, and a single-slot test before pipelining. |
| Mapping is not actually contiguous | Host look-ahead must prove physical NOC-address contiguity; otherwise use fallback. |
| More workers overload a Fabric link | Measure per-link throughput after every worker increase; use mux only where needed. |
| Core conflicts with model programs | Allocate only from CCL sub-grid and validate core availability at dispatch. |
| Ring behaves differently in full deployment | Test exact Galaxy Torus-Y topology and end-to-end model workloads, not only 1D proxy measurements. |
| FP8 gains remain poor | Attribute source/read, Fabric, and destination stages independently; do not claim a byte-reduction win from payload size alone. |

## Deliverables

1. Attribution report and reproducible isolated AG baseline.
2. Contiguous source-read and direct-contiguous-write microbenchmarks with
   correctness tests.
3. Experimental receiver-core AG program with a documented credit protocol.
4. Decision record comparing direct remote DRAM versus receiver-core DRAM
   writes, including linear and Torus-Y/ring product topology data.
5. Production fast-path implementation, fallback tests, release build, and
   Sparse MLA functional/performance comparison.

### Immediate deliverable sequence

1. Preserve the implemented connection-free receiver, completion-threshold
   repair, dtype-selected drain policy, and command-count attribution as
   reviewable commits.
2. Preserve the completed source/destination address-run histogram and the
   rejected source-only, destination-only, and combined bank-major results.
3. Preserve the completed static bank-owned prototype only as an opt-in
   attribution control.  Do not generalize its core placement or topology
   after it failed the automatic-receiver gate.
4. Freeze the accepted automatic receiver, finish generic fallback validation,
   and attribute source/Fabric production wait without changing page order,
   core count, or topology.
5. For each retained automatic-schedule change, run the 11/11 RM matrix, 20x reuse,
   release build, and 39/39 Sparse MLA correctness suite.
6. Preserve the completed receiver-confirmed Sparse MLA warm and long A/B/A on
   the target SP=4 path; use SP=2 only as historical/no-regression evidence
   because it did not expose the isolated gain.
7. Validate QuietBox/loudbox and Galaxy `FABRIC_2D_TORUS_Y`/ring, followed by
   the generic fallback regression suite.

### Concrete experiment queue

| Order | Status | Experiment | One changed variable | Pass criterion | Stop / rollback criterion |
| --- | --- | --- | --- | --- | --- |
| C0 | PASS | Current correctness baseline | Test coverage only | 11/11 RM, 20x/dtype reuse, and 39/39 Sparse MLA remain green | Any mismatch or hang; reduce with safe runner and tt-triage |
| A0 | PASS | Current command attribution | Instrumentation only | Stable samples for all role/stage counts and cycles | Counts vary unexpectedly or do not match page ownership |
| T1 | PASS | Physical-address run histogram | Observability only | Quantifies legal contiguous and same-bank runs for source/destination | No optimization until the mapping is understood |
| T2 | REJECT | Source command coalescing | Source schedule only | At least 3% median gain and lower source issue/count | Source-only FP8 was 1.870 ms; permutation cost erased command gain |
| T3 | REJECT | Bank-owned receiver drain | Destination schedule only | Lower drain issue time and at least 3% combined gain | Destination-only FP8 was 1.879 ms; combined best was 1.846 ms, below gate |
| T4 | REJECT | Two bank-owned worker pipelines | Exact-divisor/max-tail and source-local/all controls | Exact counts and correctness pass; max-tail is 14.4% / 8.2% faster inside static | Best static is still 4.1% BF16 / 1.7% FP8 slower than automatic; stop placement/fan-out work |
| P0 | PASS | SP=4 Sparse MLA perf | Model workload only | Receiver-confirmed A/B/A shows warm and long model wins with separate CCL accounting | Reopen if repeated controls fall outside the recorded A/A envelope |
| P1 | PARTIAL | Product topology | Hardware/routing only | Supported P150 8x1 ring and available P150 4x2 model path are correct and measured; actual QuietBox and Galaxy Torus-Y/ring still require recorded perf | Topology-specific hang, route/resource failure, or fallback regression |

At each row, commit or tag the exact candidate before timing it.  A failed
experiment can remain as a documented diagnostic commit, but it must not stay
enabled in the production dispatch path.

## Explicit non-goals

- Repacking MLA KV cache rows solely to make all-gather faster.
- Removing the generic scatter all-gather path.
- Assuming ring is unsupported in production.
- Increasing firmware size via `--dev` to work around a protocol or resource
  bug.
- Claiming an end-to-end model win from isolated AG measurements without a
  separately reported end-to-end measurement.
