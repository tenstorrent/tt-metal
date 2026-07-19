# All-gather Tensix receiver: implementation and iteration plan

## Purpose

This document defines the next phase of the native all-gather work for small,
row-major Sparse MLA payloads.  It is a forward-looking implementation plan,
not only an experiment log.

The design sends Fabric payloads to staging buffers in Tensix L1 instead of
using final DRAM addresses in every Fabric packet.  A receiver Tensix core then
writes the data to the correct DRAM banks.  This gives the collective control
over packet coalescing, receive buffering, DRAM-bank scheduling, NoC selection,
and receiver-core fan-out without changing the model's logical tensor layout.

The work must be developed one independently measurable change at a time.  A
more complex receiver plan is accepted only when the previous, simpler plan is
correct and profiling proves which resource is limiting it.

The existing experiment history remains in
`docs/ALL_GATHER_CONTIGUOUS_RECEIVE_PLAN.md`.  This document starts from the
latest measured receiver implementation and defines what to do next.

Status snapshot: 2026-07-18, branch
`pjosipovic/sparse-mla-new-all-gather`.  Native BF16 and FP8 batch selection,
valid-extent selection, and their combination are implemented and stable.  The
source/Fabric/credit and receiver-drain intervals are instrumented.  The
current automatic implementation uses one receiver drain RISC for BF16 and two
for FP8.  The byte, packet, window, and physical-address experiments are
complete.

The address experiment proved that logical pages rotate across eight DRAM
banks while pages eight rows apart are physically contiguous within a bank.
Source-only, destination-only, and combined per-payload bank-major
permutations reduced NOC command counts but did not pass the 3% latency gate;
their permutation work moved time into Fabric issue and receiver-produced
wait.  That code has been removed.  A **two-link static bank-owned pipeline**
was then implemented to create bank-major payloads directly without
per-payload permutation.

The host reference schedule, schedule-only kernel path, and all three command
coalescing stages are now implemented behind experimental settings.  The pure
schedule cases pass 3/3, the exact-divisor hardware cases pass for BF16 and
scaled FP8, and the source stage passes a 40/40 repeated run.  Source and local
output coalescing provide small isolated wins; receiver-drain coalescing
reduces commands exactly as intended but is off the critical path and slightly
regresses latency.  Tail-aware bank runs restore the maximum legal R12 BF16
and R20 FP8 payload sizes and pass focused correctness with exact command
counts.  Clean A/B/A is complete: max-tail improves the static path, but the
best static mode is still slower than the accepted automatic receiver for both
dtypes.  Static bank ownership is rejected for production and remains
diagnostic-only.  The active plan is to harden/genericize the automatic path,
attribute its producer/Fabric wait, then validate product topology and model
benefit.

## Executive decision

Continue with the connection-free Tensix receiver architecture.

The current measurements show:

- raw Fabric-to-L1 delivery is faster than the original remote-DRAM path;
- drain-only DRAM work is not the whole ceiling once there are enough L1
  slots, but its per-page command issue expands materially when ingress runs
  concurrently;
- one acknowledgement per payload serializes the pipeline and is the main
  reason the initial receiver was slow;
- maximum legal payload batching plus windowed credits changes the receiver
  from a large regression into a measurable win; and
- FP8 benefits more than BF16, but fixed overhead still prevents latency from
  scaling exactly with the byte reduction.

Therefore the next work is not to add Fabric clients or immediately add more
receiver cores.  MLA slice/extent semantics are complete, and interval
attribution showed that per-page NOC command issue matters.  One receiver core
using both data-movement RISCs removes about 35% from drain-only time.  In
combined mode it is retained for FP8 but not BF16.  The current accepted path
is 3.063 ms BF16 and 1.753 ms FP8 on the reference case.  FP8 additionally
uses a column-major sender/receiver core pairing; BF16 retains row-major
pairing.

The bank-major experiments now establish both sides of the decision.  A single
worker's per-payload permutation erases the command-count gain, while static
link ownership preserves physical order but loses overlap relative to the
automatic schedule.  The next bounded work therefore keeps automatic page
order, core count, and Fabric routes fixed while measuring and improving
producer/Fabric overlap.  No new Fabric connection, ERISC client, mux, or
receiver core is justified until that attribution identifies a resource that
such a change can improve.

## Current measured state

The following results were collected on the same 4x2 Blackhole mesh using
32,768 rows per device, persistent output, the realtime device profiler, a
discarded profiler warmup, and seven timed samples.

| Path | BF16, 1152 B/page | FP8 E4M3, 704 B/page | Interpretation |
| --- | ---: | ---: | --- |
| Direct remote DRAM | 3.540 ms, 31.99 GB/s | 3.373 ms, 20.52 GB/s | Production comparison baseline |
| One-slot receiver, combined | 10.704 ms, 10.58 GB/s | 7.064 ms, 9.80 GB/s | Per-payload synchronization dominates |
| One-slot receiver, L1 sink | 9.637 ms | 6.024 ms | Removing DRAM drain recovers little |
| Two-slot receiver, L1 sink | 5.979 ms | 3.736 ms | A second slot removes a large serialization bubble |
| L1 overwrite diagnostic | 2.877 ms, 39.36 GB/s | 1.749 ms, 39.57 GB/s | Raw ingress is fast; this mode is intentionally unsafe |
| Drain-only diagnostic | 1.404 ms, 80.68 GB/s | 1.270 ms, 54.48 GB/s | Destination drain plus retained source/barrier overhead; no Fabric payload or credits |
| One-RISC auto slots/window, combined | 3.071 ms, 36.87 GB/s | 2.195 ms, 31.53 GB/s | Control before the dtype-aware drain policy |
| Two-RISC drain-only | 0.915 ms, 123.73 GB/s | 0.835 ms, 82.90 GB/s | About 35% faster local drain for both dtypes |
| Two-RISC combined | 3.017 ms, 37.53 GB/s | 1.865 ms, 37.10 GB/s | BF16 +1.6%; FP8 +15.1% versus one RISC |
| Accepted dtype-aware placement | 3.063 ms, 36.98 GB/s | 1.753 ms, 39.48 GB/s | Row-major BF16; column-major FP8 sender/receiver pairs |

Relative to direct remote DRAM, the current dtype-aware automatic receiver
reduces isolated latency by about 13.5% for BF16 (3.540 to 3.063 ms) and 48.0%
for FP8 (3.373 to 1.753 ms).  The FP8 payload is 61.1% of the BF16 payload and
its accepted latency is now 57.2% of BF16 latency because its vertical core
pairing also reduces NoC contention.

Combined minus drain-only is 1.667 ms for BF16 and 0.925 ms for FP8.  These
differences are not additive kernel decompositions because ingress and drain
overlap in the combined path.  They do establish a bound: making the DRAM drain
faster cannot by itself remove the full combined latency, and the next
measurement must focus on sender/Fabric/credit work and lost overlap.  The
drain-only diagnostic still executes source reads, local sender writes, route
setup, and completion barriers; it suppresses remote Fabric payloads and
produced/consumed credit traffic, and drains the ordinary receiver L1 region to
the persistent output.  Its output contents are intentionally unspecified, so
it must never be used as a correctness mode.

The fastest safe window sizes seen so far are the maximum values allowed by
available L1 on this configuration:

- BF16: 26 slots, about 3.061 ms;
- FP8 E4M3: 25 slots, about 2.202 ms.

The automatic policy derives the slot count from available L1 and reproduces
these results.  A larger explicit request is rejected before dispatch.

### Per-RISC attribution result

Attribution is opt-in through
`TTNN_ALL_GATHER_RECEIVER_ATTRIBUTION=1`.  Each participating RISC accumulates
32-bit wrap-safe timestamp deltas and emits each aggregate once at kernel exit.
The intervals overlap across cores and RISCs; they must never be added into an
operation latency.  `scripts/analyze_all_gather_attribution.py` selects the
critical core independently for the reader, writer, and receiver roles.

The BF16 combined median moved from 3.071 ms without attribution to 3.085 ms
with attribution, an approximately 0.5% perturbation.  The shorter FP8 and
drain-only paths show more relative measurement overhead, so clean perf runs
remain the source of truth for candidate comparisons.

Critical-role medians from the 4x2, 32,768-row attribution run are:

| Dtype/mode | Operation | Critical reader | Critical writer | Critical receiver |
| --- | ---: | ---: | ---: | ---: |
| BF16 combined | 3.085 ms | 2.927 ms | 2.995 ms | 2.937 ms |
| BF16 drain-only | 1.471 ms | diagnostic only | diagnostic only | 1.360 ms |
| FP8 combined | 2.235 ms | 2.133 ms | 2.182 ms | 2.150 ms |
| FP8 drain-only | 1.320 ms | diagnostic only | diagnostic only | 1.253 ms |

The dominant receiver intervals are:

| Dtype/mode | Drain issue | Produced wait | Drain flush/other |
| --- | ---: | ---: | ---: |
| BF16 combined | 2.476 ms | 0.399 ms | 0.053 ms |
| BF16 drain-only | 1.340 ms | none | 0.019 ms |
| FP8 combined | 1.867 ms | 0.249 ms | all other receiver intervals below 0.04 ms total |
| FP8 drain-only | 1.241 ms | none | all other receiver intervals below 0.02 ms total |

Sender-side intervals still matter, but they do not point to one simple source
read fix.  In BF16 combined mode, the critical reader spends 1.380 ms in
Fabric work, 0.761 ms issuing source reads, and 0.756 ms in credit handling.
The critical writer spends 1.126 ms issuing Fabric work, 0.818 ms on its local
contribution, and 0.692 ms in credit handling.  FP8 shifts more of each sender
role into credit/local fixed cost.

Two observations determine the next experiment:

1. Drain-only issue time changes little when page bytes fall from 1152 B to
   704 B: 1.340 ms to 1.241 ms.  The receiver issues the same number of
   per-page NOC write commands, so command rate rather than payload bytes is
   the dominant drain-only cost.
2. Simultaneous Fabric ingress inflates receiver drain-issue time from 1.340
   to 2.476 ms for BF16 and from 1.241 to 1.867 ms for FP8.  This is consistent
   with NoC issue/contention and lost overlap on the receiver critical path.

The subsequent two-RISC experiment passed the full 11-case row-major matrix
and 20 repeated sliced-partial invocations per dtype.  Its clean A/B/A control
medians drifted by less than 0.1%.

With two RISCs, BF16 receiver drain issue falls from 2.476 ms to 0.813 ms, but
produced-wait rises from 0.399 ms to 2.129 ms.  FP8 drain issue falls from
1.867 ms to 0.731 ms and produced-wait rises from 0.249 ms to 1.088 ms.  The
receiver is now waiting for payload production rather than limiting it.

Decision: select the dual-RISC mechanism by default for FP8, where it passes
the 3% gate by a wide margin.  Keep one RISC for BF16 by default because its
1.6% combined improvement does not pass the written gate.  Do not add a second
receiver core; the current evidence points upstream.

The production default is now dtype-aware: `auto` selects two drain RISCs for
FP8 E4M3 and one for BF16.  Explicit `1` and `2` values remain isolated A/B
controls.  With the automatic policy, the clean reference medians are 3.060 ms
for BF16 and 1.868 ms for FP8, corresponding to 37.01 and 37.04 GB/s.

### Completed byte, packet, and window sweeps

All rows below use the two-RISC receiver so the transport ceiling is visible,
combined mode, persistent output, attribution disabled, and the same 4x2
Blackhole mesh.  Lower latency is faster.

At maximum legal row batching and the automatic maximum L1 window:

| Rows/device | BF16 latency | BF16 BW | FP8 latency | FP8 BW |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.786 ms | 36.02 GB/s | 0.493 ms | 35.12 GB/s |
| 16,384 | 1.539 ms | 36.79 GB/s | 0.960 ms | 36.03 GB/s |
| 32,768 | 3.017 ms | 37.53 GB/s | 1.865 ms | 37.10 GB/s |

This is near-linear scaling with roughly 0.04 ms fixed cost.  FP8 and BF16
converge on the same approximately 37 GB/s physical-byte ceiling, and their
slopes track the 704/1152 physical-page-byte ratio.  The earlier failure of
FP8 latency to scale with bytes is therefore resolved by the dual-RISC drain.

At 32,768 rows/device and the automatic maximum L1 window:

| Rows per Fabric payload | BF16 | FP8 |
| ---: | ---: | ---: |
| 1 | 19.639 ms | 19.328 ms |
| 4 | 5.822 ms | 5.250 ms |
| 8 | 3.703 ms | 3.132 ms |
| Maximum legal (BF16 12, FP8 20) | 3.017 ms | 1.865 ms |

The large packet-count sensitivity proves that per-payload fixed work remains
material.  Maximum legal batching is the correct baseline for the next source
coalescing experiment.

At 32,768 rows/device and maximum legal row batching:

| L1 slots/window | BF16 | FP8 |
| ---: | ---: | ---: |
| 1 | 9.939 ms | 6.454 ms |
| 2 | 5.921 ms | 3.876 ms |
| 8 | 3.500 ms | 2.171 ms |
| Automatic maximum (BF16 26, FP8 25) | 3.017 ms | 1.865 ms |

The maximum safe L1 window remains best for both dtypes.  Smaller windows
expose credit round-trip latency and are diagnostic controls, not tuning
candidates.  Consequently the next iteration must keep maximum payload
batching and maximum safe windows fixed while reducing the number of internal
source-read/L1/Fabric commands per payload.

### Current Sparse MLA validation

The complete Sparse MLA functional file passes 39/39 after persistent-output
integration with the dtype-aware automatic receiver.  Coverage includes BF16
and scaled FP8, 2x4 and 4x2 mesh orientations, sequence lengths 256 and 5120,
chunked execution, rotated cache positions, program reuse, and determinism.
The final topology-policy candidate completed through `scripts/run_safe_pytest.sh`
in 950.50 seconds with static ownership disabled and no dispatch timeout; no
tt-triage capture was needed.  Separate cold controls complete 11/11 forwards per dtype and
observe one receiver program per forward.

The final full local all-gather file also completes through the safe runner
with realtime-profiler path assertions: 66 pass and four intentional opt-in
cases skip in 66.37 seconds.  The skipped cases are the two rejected
bank-owned diagnostic controls and the two isolated performance cases already
measured separately; no ordinary correctness or selection case skips.

Same-build receiver-confirmed model measurements now use the target SP=4 x
TP=2 axis on the available P150 4x2 mesh.  A/B/A totals improve 0.35-0.47% BF16
and 4.40-5.58% FP8 for warm cache, and 3.99-5.11% BF16 and 12.13-15.69% FP8 for
long cache.  Native-CCL aggregates improve 10.34-13.76% BF16 and 36.51-42.63%
FP8.  Automatic runs each assert one receiver program; forced-direct runs
assert zero.  Program and CCL counts are identical between modes.  The exact
table and commands are recorded in
`models/demos/deepseek_v3_d_p/tests/sparse_mla/sparse_mla_all_gather_perf.md`.
Actual QuietBox and Galaxy/ring support remains necessary before product
topology sign-off.  A final `tt-smi -s` inventory check exposes eight `p150b`
boards and no QuietBox or Galaxy devices, so this host cannot satisfy that
gate.

### Latest address-attribution and coalescing result

Device-side address attribution records raw NOC destinations without changing
the production schedule.  Three stable samples of the 32,768-row reference
case show the same source mapping for BF16 and FP8:

| Observation per directional worker | Count |
| --- | ---: |
| Logical adjacent page pairs | 16,383 |
| Adjacent pairs on the same DRAM bank | 0 |
| Pages contiguous with the page eight rows earlier | 16,376 |

The output has the same eight-bank rotation.  This is the critical mapping
fact: logical order is maximally inconvenient for contiguous NOC commands,
while each bank's subsequence is a long physical run.

Three bank-major prototypes were measured and then removed from the data path:

| Prototype | BF16 median | FP8 median | Decision |
| --- | ---: | ---: | --- |
| Accepted receiver floor, A/A controls | 3.060-3.069 ms | 1.864-1.868 ms | Baseline |
| Source and destination coalesced after per-payload permutation | 3.114 ms | best 1.846 ms | BF16 regressed; FP8 gained about 1.1%, below gate |
| Source coalesced only | Not advanced after BF16 rejection | 1.870 ms | Neutral |
| Destination coalesced only | Not advanced after BF16 rejection | 1.879 ms | Regressed |

The combined FP8 prototype cut source commands from 16,384 to 6,556 and
critical receiver writes from roughly 24,000 to 9,444.  Source issue fell from
0.840 to 0.356 ms and receiver drain issue from 0.810 to 0.442 ms.  However,
writer Fabric issue rose from 0.510 to 0.697 ms and receiver-produced wait rose
from 1.027 to 1.347 ms.  The command reduction is real, but rearranging every
payload on the same worker consumes the gain.

Decision: preserve address observability, reject per-payload permutation, and
make bank order the native ownership order of the link pipeline.

### Static bank-owned command-coalescing result

The bank-owned implementation separates scheduling from the three command
sites with `TTNN_ALL_GATHER_BANK_OWNED_COALESCE=none|source|source_local|all`.
Every leg was run in a separate process with persistent output and the same
receiver protocol.  Lower latency is faster.

At the original exact-divisor payloads, R8 BF16 and R16 FP8, attribution
matched the ownership plan exactly:

| Retained stages | BF16 source | BF16 local | BF16 receiver | FP8 source | FP8 local | FP8 receiver per critical RISC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Schedule only | 16,384 | 16,384 | 49,152 | 16,384 | 16,384 | 24,576 |
| Source | 2,048 | 16,384 | 49,152 | 1,024 | 16,384 | 24,576 |
| Source + local | 2,048 | 2,048 | 49,152 | 1,024 | 1,024 | 24,576 |
| Source + local + receiver | 2,048 | 2,048 | 6,144 | 1,024 | 1,024 | 1,536 |

Clean one-variable A/B/A results were:

| Added stage | BF16 control/candidate/control | Incremental result | FP8 control/candidate/control | Incremental result |
| --- | --- | ---: | --- | ---: |
| Source | 3.764 / 3.744 / 3.762 ms | 0.5% faster | 2.082 / 2.070 / 2.083 ms | 0.6% faster |
| Local output | 3.744 / 3.717 / 3.738 ms | 0.6% faster | 2.070 / 2.046 / 2.066 ms | 1.1% faster |
| Receiver drain | 3.717 / 3.736 / 3.723 ms | 0.4% slower | 2.046 / 2.061 / 2.050 ms | 0.6% slower |

The receiver reduction is real, but its issue time is only about 0.133 ms
BF16 and 0.041 ms FP8 after coalescing.  Produced-wait remains about 2.875 ms
BF16 and 1.827 ms FP8, so the receiver is waiting for payload delivery.  The
`all` mode is therefore diagnostic; source plus local is the best exact-
divisor bank-owned configuration.

The exact-divisor helper unintentionally lowered the payload from the accepted
receiver's R12/R20 maximum to R8/R16.  Tail-aware bank runs remove that
confounder.  For 32,768 rows/device, the expected and measured directional-
worker counts are:

| Dtype | Run geometry | Source/local/Fabric commands | Aggregate receiver commands before dual-RISC partition |
| --- | --- | ---: | ---: |
| BF16 | 4 banks/link x ceil(4096/12) | 1,368 | 4,104 |
| FP8 E4M3 | 4 banks/link x ceil(4096/20) | 820 | 2,460 |

BF16 attribution measures 1,368 source, local, and Fabric commands and 4,104
receiver commands exactly.  FP8 measures 820 source, local, and Fabric
commands exactly.  The critical receiver RISC reports 1,182 writes; this is a
per-RISC value, not the aggregate 2,460, because the two drain RISCs inherit an
uneven direction/source partition.  The analysis tool must not compare that
single-RISC number with an aggregate host count.

Attribution-enabled tail runs measured 3.231 ms BF16 and 1.908 ms FP8.  These
are path-accounting samples, not acceptance timings.  Clean measurements have
now completed and separate process identities were used for both run policies.

Tail-aware `source_local` substantially improves the experimental static path:

| Dtype | Divisor A1 / max-tail B / divisor A2 | Max-tail improvement | Control drift |
| --- | --- | ---: | ---: |
| BF16 | 3.733 / 3.195 / 3.731 ms | 14.4% | 0.05% |
| FP8 E4M3 | 2.058 / 1.891 / 2.060 ms | 8.2% | 0.10% |

Receiver coalescing was then retested at maximum runs in a
`source_local/all/source_local` sequence.  It made BF16 about 0.9% slower and
FP8 about 0.7% slower, so `source_local` remains the best static variant.

The decisive production comparison used
`automatic/static-max-tail-source_local/automatic` with seven clean samples
per leg:

| Dtype | Automatic A1 | Static B | Automatic A2 | Static versus mean control | p90, B versus mean A |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 3.070 ms | 3.194 ms | 3.065 ms | 4.1% slower | 3.206 versus 3.076 ms, 4.2% slower |
| FP8 E4M3 | 1.867 ms | 1.897 ms | 1.864 ms | 1.7% slower | 1.915 versus 1.874 ms, 2.2% slower |

Automatic-control median drift was 0.16% for each dtype.  The static schedule
therefore fails G5's requirement to improve the accepted automatic receiver by
at least 3%.  It remains an opt-in diagnostic implementation; production
dispatch continues to select the automatic receiver or generic fallback.

The raw production-gate samples, in milliseconds, are retained here so the
decision does not depend on rounded table values:

- BF16 automatic A1: `3.074, 3.067, 3.070, 3.064, 3.069, 3.078, 3.073`;
- BF16 static B: `3.179, 3.180, 3.194, 3.195, 3.206, 3.183, 3.200`;
- BF16 automatic A2: `3.061, 3.065, 3.063, 3.063, 3.071, 3.074, 3.073`;
- FP8 automatic A1: `1.873, 1.867, 1.864, 1.863, 1.876, 1.872, 1.863`;
- FP8 static B: `1.896, 1.895, 1.899, 1.897, 1.893, 1.915, 1.903`; and
- FP8 automatic A2: `1.864, 1.864, 1.869, 1.870, 1.863, 1.872, 1.864`.

### Facts established so far

1. Adding a Fabric connection from the receiver can oversubscribe endpoint
   channels and hang asymmetrically.  The receiver must remain connection-free.
2. Fabric payload and its produced notification are ordered on the sender's
   existing route.
3. The receiver returns consumption locally to an associated sender core; the
   existing sender routes proxy remote credits.
4. Receiver payload buffers belong in ordinary L1.  Control semaphores belong
   in L1-small.
5. Persistent/preallocated output is required in both baseline and candidate
   measurements.
6. Maximum legal payload batching is better than 1, 2, 4, or 8 row batches for
   the measured MLA geometries.
7. Fused payload-plus-notification is better than a separate notification for
   BF16 and neutral for FP8, so fused is the default.
8. Window credit is better than per-slot credit.  One credit per completed
   slot window is the current default.
9. The full focused row-major matrix passes: 11 of 11 cases.
10. Native batch-slice, valid-extent, and combined selection pass for BF16 and
    FP8; realtime-profiler assertions prove that all six cases select the
    receiver kernel, while the fresh-output control selects the generic path.
11. Combined slice-plus-extent passes 20 consecutive invocations for each of
    BF16 and FP8.
12. Per-RISC attribution is opt-in and leaves the production path free of
    timestamp work when disabled.
13. Two receiver RISCs reduce drain-only latency by about 35% for both dtypes.
14. After dual-RISC drain, FP8 retains a 15.1% combined win while BF16 becomes
    sender/Fabric limited and retains only 1.6%.
15. Logical MLA rows rotate over eight DRAM banks; page `p + 8` is physically
    contiguous with page `p` within its bank on the measured Blackhole layout.
16. Per-payload bank-major permutation can reduce commands substantially but
    does not pass the latency gate.  Static ownership must eliminate that
    permutation rather than optimize it.
17. The two-link static schedule is a bijection for the tested 32-row and
    32,768-row shapes, and the schedule-only kernels preserve numerical output
    for BF16 and scaled FP8 without a hang across 40 repeated cases.
18. One raw contiguous source read per bank run is numerically correct and
    stable; it gives only about a 0.5-0.6% isolated latency win.
19. Coalescing the local output write adds another 0.6-1.1% win.  Coalescing
    the receiver drain cuts commands by 8x/16x but regresses total latency by
    0.4-0.6% because receiver issue is not critical.
20. Tail-aware R12/R20 runs are correct and their source/local/Fabric counts
    exactly match 1,368 BF16 and 820 FP8 commands per directional worker.
21. Max-tail runs improve the static bank-owned path by 14.4% BF16 and 8.2%
    FP8 compared with exact-divisor runs; reduced payload size was a real
    confounder in the first static measurements.
22. Coalescing receiver writes at maximum runs still regresses combined
    latency.  Receiver command issue is not the active limiter.
23. The best static path is 4.1% slower BF16 and 1.7% slower FP8 than the
    accepted automatic receiver, with only 0.16% control drift.  Static bank
    ownership is rejected for production despite its exact command reduction.
24. Inactive experimental compile-time paths must use nonzero sentinel values
    for divisor/modulo operands.  A zero sentinel caused the generic receiver
    kernel to fail JIT compilation even though the static lambda was unused;
    using `1` preserves the inactive path without changing its schedule.

### Current limitations

The current receiver predicate is deliberately narrow:

- row-major rank-4 input;
- gather dimension 2;
- shape dimensions 0 and 1 equal to one;
- one active mesh axis;
- linear topology;
- no more than four devices on that axis;
- row-major interleaved persistent output for the receiver path;
- the two-RISC receiver is experimental and restricted to combined or
  drain-only mode with window credits; and
- no Galaxy ring/Torus-Y receiver enablement yet.

The prototype now exercises the Sparse MLA cache-selection semantics in the
focused collective test.  Model integration and the Galaxy ring/Torus-Y
deployment still require explicit sign-off; neither may be inferred from the
4x2 linear result.

## Target data path

The desired steady-state pipeline is:

```text
source DRAM
    |
    | local NoC read, coalesced when physical mapping permits
    v
source Tensix L1
    |
    | existing Fabric sender route
    | packet destination = receiver Tensix L1 slot
    v
destination receiver L1 slot
    |
    | one or two receiver data-movement RISCs
    | bank-aware local NoC writes
    v
persistent destination DRAM output
```

The Fabric packet's final NOC destination is a Tensix L1 address.  The ERISC
does not need a new logical connection to the receiver.  It delivers the
packet through the normal Fabric-to-NoC path, and the selected Tensix core owns
the subsequent DRAM writes.

This architecture creates two independent scheduling domains:

1. **transport scheduling** decides which source rows are combined into a
   payload, which destination receiver and L1 slot receive it, and when the
   slot window can be reused; and
2. **DRAM scheduling** decides which receiver RISC/NoC writes each output page
   and in what order.

They must remain separately measurable.  A transport improvement should not
be hidden by changing the DRAM page order in the same commit.

### Concrete two-link bank-owned pipeline

The first new implementation is intentionally bounded to the measured 4x2
case with two active links and eight interleaved DRAM banks.  Existing sender
and receiver cores are paired by link:

```text
link 0 sender core  ---- existing Fabric route ---->  link 0 receiver core
  owns banks 0,2,4,6                                drains banks 0,2,4,6

link 1 sender core  ---- existing Fabric route ---->  link 1 receiver core
  owns banks 1,3,5,7                                drains banks 1,3,5,7
```

No packet is permuted in L1.  A sender enumerates one owned bank at a time,
reads a physically contiguous run into one CB entry, writes its local output
in the same schedule, and sends that CB entry to the paired receiver L1 slot.
The receiver derives the same bank/run from the monotonically increasing batch
sequence and drains the complete run with one local NOC write.

For `B` DRAM banks, `L` links, link index `l`, and a fixed batch of `R` rows,
the first logical page of batch `q` in owned-bank slot `k` is:

```text
bank = l + k * L
first_page = selected_page_start + bank + q * R * B
```

The schedule is valid only when:

- `B % L == 0`;
- selected source start, source-stripe size, and each source's output-stripe
  offset preserve the bank mapping;
- each full run contains `R` rows and the final run contains
  `min(R, pages_per_bank - run_start)` rows;
- every page in the run has the same NOC destination X/Y and consecutive raw
  byte addresses;
- source and output page sizes match and the complete run fits one legal
  Fabric payload and one receiver slot; and
- every selected page is owned exactly once across links.

For the current 32,768-row case, each bank contains 4,096 rows.  The
exact-divisor control uses `R=8` for BF16 (9,216 B) and `R=16` for FP8
(11,264 B).  The tail-aware candidate uses the maximum legal `R=12` BF16
(13,824 B) and `R=20` FP8 (14,080 B).  The expected source command count per
directional link worker is therefore:

| Dtype | Per-page commands | Exact-divisor control | Max-run tail candidate |
| --- | ---: | ---: | ---: |
| BF16 | 16,384 | 2,048 | 1,368 |
| FP8 E4M3 | 16,384 | 1,024 | 820 |

For three remote sources, a receiver link should issue 4,104 BF16 or 2,460 FP8
run writes before any dual-RISC subdivision, rather than one write per row.
Instrumented aggregate counts must match the host plan exactly.  Per-RISC
counters must also record the ownership split so an imbalanced critical RISC
is not mistaken for missing work.  Lower latency without matching ownership
and counts is not accepted as evidence.

The first version is opt-in and fails host validation when any precondition is
not met.  Production `auto` mode continues to use the accepted receiver or
generic fallback.  Tail batches are implemented for the bounded aligned case.
Sliced offsets that change bank alignment, other bank counts, other link
counts, ring direction partitioning, and generalized core placement remain
follow-up steps, not assumptions hidden in the prototype.

## Protocol invariants

Every implementation and optimization must preserve these invariants:

1. A payload slot is owned by exactly one source/sequence until consumption is
   acknowledged.
2. Payload data becomes visible before the corresponding `produced` sequence.
3. A receiver drains only a sequence it has observed as produced.
4. `consumed(window)` is published only after every payload in that complete
   or final partial window is safe to overwrite.
5. A source reuses a window only after it has received the required credits
   from all destinations.
6. Sender reader and writer routes proxy disjoint destination subsets.  Their
   union contains every remote destination exactly once.
7. Initialization and final completion have separate thresholds.  Local
   receiver-reset acknowledgements are initialization events, not remote
   completion events.
8. A receiver opens no Fabric connection and owns no additional ERISC client
   channel.
9. Two receiver RISCs or cores never write the same output page or cache line.
10. The host proves L1 bounds, packet bounds, core availability, topology, and
    output-page ownership before device dispatch.

## Buffer and semaphore contract

### Persistent output

All performance and model tests must provide a preallocated DRAM output tensor.
The output address can change through program-cache overrides, but allocation
and output creation must not be included in the timed collective.

The correctness tests must prefill the output with a non-result sentinel for
partial gather cases.  They must verify both the selected region and every
untouched region.  This catches an incorrectly widened receiver drain.

### Receiver staging

Receiver payload storage uses ordinary worker L1.  For each receiver core:

```text
receiver_buffer_base
    + source_index * slots_per_source * slot_stride
    + sequence_slot * slot_stride
```

`slot_stride` is aligned to all Fabric and local NoC requirements.  Host code
computes the largest safe slot count from the real L1 end address after kernel,
CB, and runtime allocations.  It must not assume that every architecture or
firmware build exposes the same L1 interval.

### Control state

Semaphores are allocated in L1-small when the architecture exposes that memory:

- local sender credit sequence;
- local receiver-consumed sequence;
- one produced sequence per source stream;
- receiver initialization acknowledgement; and
- final completion state where the existing collective requires it.

No large payload is stored in L1-small.  Tests should inspect program metadata
or allocation addresses so this remains a checked contract rather than a
comment-only convention.

## Core allocation plan

Core allocation is part of the performance design.  Picking the next free core
is acceptable for the first correctness prototype but not for the production
policy.

### Roles

For every active Fabric link, the program may allocate:

- one existing sender core, which owns the Fabric route;
- one primary receiver core, which owns L1 staging and DRAM drain; and
- optionally one secondary receiver core, only after one-core profiling shows
  a bottleneck that a second core can address.

Receiver cores must come from the operation's legal CCL worker sub-grid.  They
must not overlap Fabric router cores, other operation workers, reserved dispatch
cores, or model compute allocations.

For the first bank-owned experiment, freeze the current physical core choices.
This ensures the measurement changes only page ownership and NOC command
shape.  Each link's existing sender core performs source DRAM download on its
reader data-movement RISC and local-output writes/Fabric injection on its
writer RISC.  Its already-paired receiver core performs remote-output DRAM
writes.  The sender-to-receiver L1 target is encoded in the Fabric packet
header exactly as in the accepted receiver path.

Only after that schedule wins should placement become a measured variable.
The placement experiment must enumerate legal pairs rather than hand-pick one
apparently close core.  For every candidate pair it records:

- sender and receiver logical and physical coordinates;
- link identity and owned DRAM banks;
- source-read NOC and receiver-write NOC;
- hop counts to each owned DRAM bank;
- L1 base, end, slot count, and unused headroom;
- overlap/conflict checks against router, dispatch, CCL, and model cores; and
- isolated source issue, receiver issue, Fabric wait, and total latency.

Do not hard-code a bank-to-NOC preference from topology diagrams.  Select it
only after the isolated counters show a repeatable benefit, because Fabric
ingress and DRAM drain can contend for the same on-chip path.

### Placement scoring

At program creation, enumerate the physical DRAM destinations for one complete
steady-state window and score each legal receiver core using:

1. bytes and command count reachable efficiently through NOC0 and NOC1;
2. hop count to the active DRAM banks;
3. balance between the two proposed NoC page lists;
4. available ordinary L1 after all program allocations;
5. proximity to the Fabric ingress path when this affects NoC contention;
6. deterministic mirrored placement on every device; and
7. absence of overlap with the sender or model core plan.

The score and selected page ownership should be printable under a debug log
level and capturable by tests.  If no legal plan exists, select direct scatter
before dispatch.

### Page ownership

Start with one receiver core and one data-movement RISC.  Then evaluate, in
order:

1. one core, bank-grouped page order;
2. one core, two RISCs/NoCs with disjoint page lists;
3. two cores, one RISC each with disjoint page lists; and
4. two cores, two RISCs each only if the first three stages show remaining
   receiver-side headroom.

The host should generate a page-owner table.  Validation must prove that every
selected output page appears exactly once and every unselected output page
appears zero times.

## Goals and completion criteria

Goals are ordered gates.  A performance result cannot waive a failed semantic,
protocol, or topology goal.

| ID | Goal | Completion criterion |
| --- | --- | --- |
| G0 | Reproduce the current floor | Direct and automatic receiver BF16/FP8 results reproduce within 3% median drift in an interleaved A/B/A session. |
| G1 | Prove buffer semantics | Persistent output is used, payload is in ordinary L1, semaphores are in L1-small, and full/tail output is correct for BF16 and FP8. |
| G2 | Support MLA selection | Receiver mode passes `batch_slice_idx`, `valid_gather_extent`, and their combination, including sentinel checks for untouched output. |
| G3 | Stabilize the protocol | Each required focused BF16 and FP8 case passes 20 consecutive program-reuse iterations with no hang, stale epoch, reset, or tt-triage failure. |
| G4 | Account for remaining time | Direct, L1-overwrite diagnostic, safe L1 sink, drain-only, combined timing, and per-RISC intervals identify the next critical-path experiment. |
| G5 | Eliminate per-page commands without permutation | A static bank-owned link schedule reduces the expected source/local/receiver NOC command counts and improves the current automatic median by at least 3%. |
| G6 | Allocate cores generically | Host placement validates legal sender/receiver pairs, L1 capacity, bank ownership, NoC reachability, and deterministic mirrored plans on every supported device. |
| G7 | Use two receiver data-movement RISCs only when useful | Dual-RISC drain improves combined latency by at least 3% with no correctness or p90 regression.  Otherwise retain one RISC. |
| G8 | Fan out only when useful | A second receiver core improves combined latency by at least 5% after synchronization and L1 cost.  Otherwise retain one receiver. |
| G9 | Preserve the generic collective | Ineligible mappings select the existing path before dispatch and regress by no more than 3%. |
| G10 | Validate product topology | Required QuietBox configurations and Galaxy linear/ring/Torus-Y configurations pass correctness and have separate performance records. |
| G11 | Deliver model benefit | Sparse MLA correctness passes and measured end-to-end speedup is consistent with separately reported AG time saved. |

### Current goal status

| Goal | Status on 2026-07-18 | Evidence or next action |
| --- | --- | --- |
| G0 | Passed on the reference 4x2 setup | Current automatic receiver is 3.060 ms BF16 and 1.868 ms FP8; preserve interleaved A/B/A controls |
| G1 | Passed for focused receiver cases | Persistent output is used; payload slots are ordinary L1; control semaphores are L1-small; a 64 B negative test proves automatic fallback and forced-mode rejection before dispatch |
| G2 | Passed | Six native selection cases pass with profiler path assertions; fresh output remains a fallback control |
| G3 | Passed for combined BF16/FP8 selection | Each combined slice-plus-extent case passes 20 consecutive safe-runner invocations |
| G4 | Passed for the next decision | Address attribution proves eight-bank rotation and same-bank physical runs; permutation prototypes expose where their command-count gain is lost |
| G5 | Rejected for production | Max-tail beats the divisor static control by 14.4% BF16 / 8.2% FP8, but the best static path is still 4.1% / 1.7% slower than automatic; retain only as an opt-in diagnostic |
| G6 | Passed for the accepted receiver plan | Reference 4x2 and supported 8x1 ring plans validate legal cores, ordinary-L1 payload bounds, and configured L1-small control capacity; rejected static bank ownership is not generalized |
| G7 | Passed for FP8; rejected as BF16 default | 40/40 reuse and 11/11 row-major cases pass; FP8 combined improves 15.1%, BF16 only 1.6% |
| G8 | Not started and not justified | Current receiver is producer/Fabric limited; do not add a receiver core without new attribution showing receiver-side headroom |
| G9 | Passed on the reference 4x2 and supported 8x1 ring setups | Production `auto` passes 9/9 explicit receiver/direct selections, 11/11 row-major mappings, tiled and L1-sharded fallbacks, configured-capacity fallback, effective-range routing, and same-process policy-cache switching; product-topology coverage remains G10 |
| G10 | Partial | Supported P150 8x1 1D ring is correct and measured; actual QuietBox and Galaxy linear/ring/Torus-Y remain |
| G11 | Passed on available P150; product sign-off pending G10 | Post-integration Sparse MLA is 39/39 and receiver-confirmed SP=4 A/B/A records CCL and end-to-end gains; actual QuietBox/Galaxy measurement remains |

### Performance targets

The current automatic receiver result is the floor for future candidates:

- BF16 reference median is 3.060 ms; the 3% keep threshold is 2.968 ms;
- FP8 reference median is 1.868 ms; the 3% keep threshold is 1.812 ms;
- no accepted variant may regress p90 by more than 3%; and
- every new core, slot, queue, or synchronization mechanism must have a
  positive measured incremental value.

These numbers are reference-hardware gates, not portable product KPIs.  On a
different box, first establish that box's own direct and automatic baselines.

For FP8 scaling, report the fit:

```text
T = T_fixed + num_packets * T_packet + num_windows * T_window + bytes / BW
```

The goal is either to bring `T(FP8) / T(BF16)` closer to the physical byte
ratio or to identify the irreducible fixed term with measured evidence.  Do
not claim scaling solely from effective bandwidth, because bandwidth can rise
when latency remains dominated by fixed work.

For the active dual-RISC experiment, compare clean, attribution-disabled runs
against the frozen one-RISC medians.  The acceptance gate is at least 3%
combined median improvement for the affected dtype, no more than 3% p90
regression, and the complete focused correctness/stability ladder.  A
drain-only improvement without a combined improvement is diagnostic evidence,
not sufficient reason to enable the extra RISC.

## TDD iteration loop

Every iteration follows the same sequence:

1. Write one hypothesis and name the metric expected to change.
2. Add or tighten the focused test before changing the implementation.
3. Make one architectural or tuning change.
4. Build with `./build_metal.sh --release`.
5. Run the smallest correctness selector through `scripts/run_safe_pytest.sh`.
6. Run the relevant 20-iteration stability selector.
7. Run isolated interleaved A/B/A measurements in separate processes.
8. Compare median, minimum, p90, raw samples, and effective payload bandwidth.
9. Keep, revise, or disable the change using the predefined gate.
10. Record the result and make the iteration a standalone commit.

Do not combine any two of the following in one measured iteration:

- protocol changes;
- batch-size changes;
- slot/window changes;
- DRAM page reordering;
- NoC ownership changes;
- receiver-core placement changes;
- receiver-core count changes; or
- topology/routing changes.

Environment-controlled experiments must run in separate processes because the
compiled program cache must not reuse a program created under a different
experimental policy.

## Ordered implementation phases

### Phase 0: freeze and validate the current candidate

**Status: complete on the reference 4x2 Blackhole setup.**

**Hypothesis:** The automatic max-slot/window policy is a stable floor and its
dispatch predicate behaves deterministically.

Tasks:

1. Run the new sliced/partial fallback tests normally.
2. Run them with the realtime profiler and assert that the receiver kernel is
   absent while fallback support remains intentional.
3. Run the full focused row-major matrix.
4. Repeat BF16 and FP8 receiver cases 20 times.
5. Re-run direct and automatic BF16/FP8 A/B/A measurements.
6. Record exact commit, hardware identity, routing, slot count, packet rows,
   warmups, samples, and raw profiler artifacts.

Exit: G0, G1, and the current fallback portion of G3/G9 pass.

### Phase 1: add native partial and sliced receive support

**Status: complete for focused BF16 and FP8 cases.**

**Hypothesis:** MLA selection changes only the source page range and output
page ownership; it does not require a different transport protocol.

Tasks:

1. Extend the host eligibility proof to include `batch_slice_idx`.
2. Generate source page offsets from the selected batch without changing the
   persistent output allocation.
3. Extend the receiver page-owner table to `valid_gather_extent`.
4. Handle the final partial transport batch and final partial credit window.
5. Verify the leading selected rows and sentinel-filled untouched rows.
6. Add combined batch-slice plus valid-extent coverage.
7. Force receiver mode in focused tests while retaining all safety checks.

Keep batch-slice-only, valid-extent-only, and combined support as separate
commits so an addressing bug has a small review surface.

Exit: G2 and G3 pass.  This phase is a prerequisite for claiming Sparse MLA
integration performance.

### Phase 2: complete stage attribution

**Status: complete.  Drain-only, interval attribution, and independent byte,
packet, and window sweeps are complete.**

**Hypothesis:** After window credits remove round-trip serialization, the
remaining gap is a mix of per-packet receiver scheduling and DRAM command
issue, not raw Fabric bandwidth.

Measure, without changing host dispatch shape:

1. direct remote DRAM;
2. unsafe L1 overwrite, only as a raw-ingress upper bound;
3. safe L1 sink with real produced/consumed sequencing;
4. drain-only using the exact production receiver page list;
5. combined ingress plus drain.

Sweep at least three total byte counts and two packet/window counts for BF16
and FP8.  Fit the timing model above and report residual error.  Add lightweight
kernel counters or timestamps for:

- sender source-read time;
- Fabric send/credit-stall time;
- receiver produced-wait time;
- receiver local NoC issue/flush time; and
- final barrier time.

Exit: G4 passes and names the next bottleneck.  If combined time is already at
the safe-ingress limit, skip DRAM fan-out and work on packet/credit scheduling.

The first drain-only calibration is:

| Mode | BF16 | FP8 E4M3 |
| --- | ---: | ---: |
| Drain-only | 1.404 ms | 1.270 ms |
| Combined | 3.071 ms | 2.195 ms |
| Non-additive difference | 1.667 ms | 0.925 ms |

This result moves source/Fabric/credit attribution ahead of receiver DRAM
schedule work.  Do not interpret the difference as a serial phase because the
production path overlaps ingress and drain.

The subsequent interval counters refined that decision.  Receiver drain issue
occupies 2.476 ms of the BF16 combined critical receiver and 1.867 ms of the
FP8 critical receiver.  Drain-only issue is 1.340 ms and 1.241 ms respectively.
The near byte-insensitivity of drain-only and the combined-path inflation make
receiver NOC command issue/ingress contention the first architecture target.
The subsequent size, packet, and window sweeps established a shared
approximately 37 GB/s physical-byte ceiling at maximum batching/window size,
and exposed a large per-payload and per-window cost at smaller settings.  The
address and permutation experiments then showed that static bank ownership,
not another per-payload rearrangement, is the next bounded experiment.

### Phase 3: implement the two-link bank-owned pipeline

**Status: complete and rejected for production.**  Instrumentation,
prerequisite sweeps, the pure host plan, schedule-only ownership, all command
coalescing stages, aligned maximum-run tails, exact attribution, and clean
A/B/A measurements are complete.  Max-tail fixes the divisor payload
confounder, but the best static schedule remains slower than automatic.

**Hypothesis:** Assigning complete DRAM-bank runs to existing link pipelines
will preserve physical order end to end, eliminate per-payload permutation,
and convert per-page source/local/receiver NOC commands into one command per
run without reducing Fabric overlap.

Completed prerequisite iterations:

1. Add per-RISC timestamps or counters around source read, route write,
   produced notification, credit wait, and final barrier.  **Done.**
2. Confirm that counters add no more than 1% to the measured median; otherwise
   use them only in attribution builds and keep production measurements clean.
   **Done for BF16 combined at about 0.5%; all acceptance perf remains
   attribution-disabled because shorter modes have higher relative overhead.**
3. Sweep bytes at fixed packet/window count, then packets at fixed bytes, then
   windows at fixed packets.  Change only one independent variable per run.
   **Done: three byte sizes, four packet sizes, and four window sizes per
   dtype.**
4. Measure sender-side batching up to the legal Fabric packet size and verify
   whether nominally coalesced rows are emitted as one packet or multiple NOC
   transactions.  **Done.**
5. Record source and destination physical-address adjacency, same-bank runs,
   and command counts.  **Done: logical pages rotate across eight banks.**
6. Try source-only, destination-only, and combined bank-major permutation.
   **Done and rejected: command counts improved but total latency did not pass
   the 3% gate.**

Active TDD iterations, each retained as a standalone commit only if its gate
passes:

1. **Host plan only — complete.** Add a pure schedule builder/reference
   enumerator for `(num_banks, num_links, selected_start, selected_pages,
   rows_per_run)`.
   Prove no duplicates, no holes, in-range pages, and identical source/local/
   receiver batch order.  This commit must not change kernel timing.
2. **Schedule-only kernel path — complete.** Behind an explicit experimental
   switch, make both link pipelines enumerate bank-owned pages but continue
   issuing one source read, local write, and receiver write per page.  This
   isolates ownership and protocol correctness from command coalescing.  It
   must pass focused BF16/FP8 output, path selection, and 20x reuse before
   timing.
3. **Coalesced source read — complete.**
   Replace the `R` per-page reads for one bank run with one raw contiguous read
   after debug validation confirms identical NOC X/Y and consecutive byte
   addresses.  Exact-divisor counts are 2,048 BF16 and 1,024 FP8 per
   directional worker.  It passes 40/40 stability and adds about 0.5-0.6%.
4. **Coalesced local output write — complete.** Keep the source result from step 3 and
   replace the sender's `R` local-output writes with one same-bank run write.
   Remote receiver writes stay per-page.  Counts match and this adds about
   0.6% BF16 and 1.1% FP8.
5. **Coalesced receiver drain — measured and rejected for latency.** Replace each receiver's `R` per-page writes
   with one run write.  Verify exact remote-source slab offset and bank
   identity before enabling the raw command.  The command reduction is exact,
   but clean total latency regresses 0.4-0.6%; keep it disabled in the current
   best candidate unless the max-run tail comparison changes the critical
   path.
6. **Maximum-run tails — complete.** The host reference and all kernels use
   ceiling run counts and explicit final-run lengths.  Focused BF16/FP8 and
   exact R12/R20 source/local/Fabric counts pass.  Clean
   divisor/max-tail/divisor shows 14.4% BF16 and 8.2% FP8 improvement inside
   the static path.
7. **Receiver coalescing at maximum runs — complete and rejected.** It is
   0.9% slower BF16 and 0.7% slower FP8 than `source_local`.
8. **Automatic/static/automatic production gate — complete and rejected.**
   Static is 4.1% slower BF16 and 1.7% slower FP8 than the mean automatic
   control.  Do not proceed to placement or topology generalization for this
   schedule.

At every kernel step, counters must show the expected command-count delta and
all unrelated role counts must remain stable.  If schedule-only is already
slower by more than 3%, first attribute the extra address arithmetic; do not
hide it by coalescing several stages at once.

Exit: the rejection branch is complete.  Counter agreement is exact and
correctness is green, but G5 does not pass.  The schedule changes sender/Fabric
production order enough that its command reduction does not preserve the
accepted automatic pipeline's overlap.  Receiver write coalescing cannot
repair this because the receiver is already waiting for payload production.

### Phase 4: conditionally optimize one receiver's DRAM schedule

**Status: closed for the current critical path.**  Per-payload page reordering
and static receiver-run coalescing both reduce commands but fail combined
latency gates.  Attribution shows the receiver waiting on production, so no
additional receiver scheduling work is justified now.

**Hypothesis:** Interleaved output pages can be drained with fewer bubbles when
ordered by physical DRAM bank and selected for the receiver core's NoC view.

**Prerequisite:** Phase 2/3 counters show receiver local-NoC issue, flush, or
destination-bank pressure on the combined critical path.  The current
drain-only result alone does not satisfy this prerequisite.

Iterations:

1. Preserve the exact page set but group consecutive writes by DRAM bank.
2. Compare stable bank rotation against logical page order.
3. Coalesce only physically consecutive pages into a larger local NOC write.
4. Move the receiver core among legal candidates using the placement score.
5. Compare source-core and receiver-core pairings for NoC contention.

Each step first runs drain-only.  It proceeds to combined timing only if
drain-only improves by at least 3%.  A drain-only win that does not change
combined time means the pipeline is limited elsewhere and the scheduling
change should not add production complexity.

Exit: one receiver is proven not to be the limiter for the best measured
schedule.  Reopen this phase only when a retained upstream change moves at
least 3% of combined time back into receiver issue/drain intervals.

## Post-T4 plan: harden automatic receiver, then attack producer/Fabric wait

The static experiment changes the next iteration order.  The accepted
automatic receiver is the production candidate.  The bank-owned path is useful
for attribution, but it is not the base for core placement, fan-out, topology
generalization, or model integration.

### New goals

| ID | Goal | Completion criterion |
| --- | --- | --- |
| N0 | Freeze a green automatic candidate | Release build, 11/11 row-major cases, 39/39 Sparse MLA cases, and BF16/FP8 isolated controls pass from one revision with static ownership disabled |
| N1 | Prove generic dispatch safety | Every receiver eligibility term is host-derived; ineligible fresh, tiled, sharded, nonpersistent, unsupported-axis, and unsupported-topology cases select fallback before JIT/dispatch |
| N2 | Attribute the accepted producer/Fabric ceiling | At least 90% of the automatic critical interval is assigned to source issue, local contribution, Fabric issue/backpressure, receiver produced-wait, receiver drain, or barriers for both dtypes |
| N3 | Improve the accepted schedule without ownership reorder | One isolated change improves either dtype median by at least 3%, does not regress the other dtype or p90 by more than 3%, preserves operation count, and passes N0/N1 gates |
| N4 | Validate deployed routing and core resources | QuietBox/loudbox and Galaxy linear/ring/Torus-Y each have a host-valid core/L1/route plan, correctness/stability record, path-selection proof, and direct/automatic perf table |
| N5 | Demonstrate attributable model benefit | Target SP=4 and production SP/ring Sparse MLA runs report AG calls, AG time saved, total latency, and the explained versus residual end-to-end delta |

### Iteration N0: validation checkpoint

1. Build with `./build_metal.sh --release` and no `--dev`.
2. Run the 11-case row-major matrix in production `auto`, not
   `force_receiver`, so intentionally ineligible cases exercise fallback.
3. Run the six slice/extent receiver-selection cases with realtime-profiler
   kernel-source assertions and the fresh-output fallback control.
4. Repeat the combined slice/extent BF16 and FP8 cases 20 times.
5. Run all 39 Sparse MLA functional cases through the safe runner.
6. Reproduce automatic BF16 and FP8 medians within 3% of the 3.060/1.868 ms
   reference envelope.  Record raw samples and p90.

Any failure in this checkpoint blocks performance iteration.  A compile error
is not classified as a hang.  A timeout must be reduced and diagnosed from the
safe-runner tt-triage artifact before changing firmware size or timeout policy.

### Iteration N1: dispatch and resource proof

Replace shape-name knowledge with an explicit receiver plan record containing:

- input/output physical page geometry and selected gather range;
- persistent output requirement and output allocation identity;
- payload bytes, slot stride, slot count, ordinary-L1 base/end, and headroom;
- L1-small semaphore addresses and counts;
- sender and receiver logical/physical cores and active NoCs;
- Fabric route identity, direction, topology, and link ownership; and
- one rejection reason when the receiver is not legal.

Add host-only unit tests for the plan and hardware tests that prove receiver or
fallback selection.  The plan must use a nonzero safe sentinel for every
inactive compile-time divisor/modulo operand so unsupported paths still JIT
compile.  Environment-only experiments remain out of production policy or are
part of program identity.

**Reference 4x2 status: complete on 2026-07-18.**  Automatic dispatch now
builds a host-side `ReceiverL1Plan` before allocating semaphores or creating a
program.  The plan validates persistent output, rank/layout/gather mapping,
batch and extent selection, linear topology/device count, DRAM/interleaved
input and output, matched page geometry, Fabric payload size, two legal worker
cores per link, ordinary-L1 base/end, payload rows, and slot-window capacity.
Every rejection has a concrete debug reason; `force_receiver` reports that
reason as a host fatal, while `auto` selects direct multicast.

All receiver experiment controls are decoded once into
`AllGatherReceiverPolicy` in the operation arguments and included in the
program hash.  A same-process `auto -> force_direct -> auto` test observes
receiver -> direct -> receiver kernel sources and correct output, proving that
different policies cannot collide in the program cache.

The expanded evidence is:

| Gate | Result |
| --- | --- |
| Release build | PASS with `./build_metal.sh --release` |
| Full focused CCL file | 33 PASS / 4 intentional opt-in skips |
| Receiver/direct selection matrix | 9/9 PASS with realtime-profiler kernel-source assertions |
| Row-major mapping matrix | 11/11 PASS with receiver/direct path assertions |
| Tiled DRAM fallback | PASS, native all-gather observed and receiver absent |
| Row-major L1 height-sharded fallback | PASS, native all-gather observed and receiver absent |
| Two-core sub-grid | PASS through direct multicast; receiver rejected before program creation |
| Oversized 256-slot request | PASS through direct multicast; receiver rejected by ordinary-L1 bound |
| Full multi-batch direct gather | PASS after limiting the pages-per-batch assertion to sliced/partial gathers |
| Slice-plus-extent reuse | 20/20 BF16 and 20/20 FP8 PASS with receiver path assertions |
| Isolated automatic perf | 3.062 ms BF16 / 1.864 ms FP8; p90 3.077 / 1.870 ms |

The post-change Sparse MLA validation also passes 39/39 through
`scripts/run_safe_pytest.sh` in 951.50 seconds with static ownership disabled.
It covers BF16 and scaled FP8, 2x4 and 4x2 mesh orientations, short and long
sequence lengths, chunking, rotation, indexer reuse, and determinism.  There
was no dispatch timeout and no tt-triage capture was needed.  This completes
N1 on the reference 4x2 setup.

**Supported 8x1 ring extension: complete on 2026-07-18.**  The accepted
receiver reuses the existing 1D ring routes and supports up to eight devices.
On an 8-chip P150 Blackhole system with 32,768 rows/device and seven release
samples, forced receiver changed BF16 from 4.363 to 5.531 ms (26.8% slower)
and FP8 from 3.890 to 3.245 ms (16.6% faster).  Automatic routing therefore
keeps direct/unicast for BF16 and selects the receiver only for FP8 at an
effective selected height of at least 32,768 pages.  An allocated 32,768-row
FP8 cache with `valid_gather_extent=128` selected direct, proving that policy
uses the active range rather than unused allocation height.

The ring decision is now bounded to that measured configuration.  The raw
`FabricConfig` is part of `AllGatherParams` and therefore the program-cache
identity; automatic receiver selection applies its FP8 crossover only to an
eight-device `FABRIC_1D_RING`.  It does not extrapolate the result to a
different ring size or to `FABRIC_2D_TORUS_Y`, even though both expose a Ring
active-axis topology.  Forced receiver remains available for bring-up.

A 4x2 P150 Torus-Y proxy control timed out for both forced receiver and forced
direct at 128 BF16 rows/device.  `scripts/run_safe_pytest.sh` captured
`generated/tt-triage/triage.csv` and reset the system after each hang.  Because
the established direct multicast control also hangs, this is a P150
Torus-Y/control-path limitation rather than an isolated receiver failure; it
does not replace the required actual-Galaxy test.  The rebuilt 4x2 linear auto
path and measured 8x1 FP8 ring auto path both passed afterward.

The receiver resource plan counts the global barrier and every receiver
control semaphore before common-barrier allocation.  Payload is ordinary L1;
controls require 112 B for four-device BF16, 176 B for eight-device BF16, and
192 B for eight-device dual-RISC FP8 at 16-byte alignment.  With only 64 B
configured, automatic mode falls back and remains correct while forced mode
reports `control semaphores exceed` before device dispatch.  This proves
configured structural capacity, not transient allocator free capacity.

Exact commands, raw ring samples, p90 values, and the small-transfer crossover
are recorded in `docs/ALL_GATHER_CONTIGUOUS_RECEIVE_PLAN.md`.  This is supported
1D-ring evidence, not a substitute for the actual Galaxy Torus-Y or QuietBox
N4 gates.

### Iteration N2: producer/Fabric attribution on the accepted order

Keep page order, receiver placement, packet rows, slots, credit window, and
topology fixed.  Collect exact counts and wrap-safe intervals for:

1. source DRAM read commands and issue cycles;
2. local-output commands and issue cycles;
3. Fabric payload commands, bytes, and issue/backpressure cycles;
4. credit-window waits and completion barriers;
5. receiver produced-wait; and
6. receiver drain issue/flush.

Run attribution for 8K, 16K, and 32K rows/device and for BF16/FP8.  Fit latency
against payload bytes, payload count, and credit windows.  Do not sum
overlapping RISC intervals.  The critical worker/RISC owns each comparison.

The decision is binary:

- if receiver drain becomes at least 30% of critical time, reopen receiver
  placement/dual-NoC work;
- otherwise, keep one receiver core and work only on source/Fabric production
  and overlap.

**Reference 4x2 status: complete on 2026-07-18.**  All six points ran through
the safe pytest runner on the final N1 release build, with production
`auto`, maximum legal payload rows, automatic slots, fused notification,
window credit, and static ownership disabled.  Clean latency uses attribution
disabled; stage intervals use a separate three-sample attribution run.

| Rows/device | BF16 clean median / p90 | FP8 clean median / p90 |
| ---: | ---: | ---: |
| 8,192 | 0.800 / 0.801 ms | 0.499 / 0.511 ms |
| 16,384 | 1.556 / 1.559 ms | 0.957 / 0.965 ms |
| 32,768 | 3.069 / 3.070 ms | 1.867 / 1.869 ms |

A least-squares fit against rows/device gives an approximately 0.044 ms fixed
cost for both dtypes.  The BF16 slope is 0.756 ms per 8,192 rows and the FP8
slope is 0.456 ms per 8,192 rows.  The slope ratio is 60.3%, close to the
704/1,152 physical-page-byte ratio of 61.1%.  Effective receive bandwidth at
32K is 36.90 GB/s BF16 and 37.07 GB/s FP8.

The attribution-enabled operation and critical-role medians are:

| Dtype | Rows/device | Operation | Reader | Writer | Receiver | Longest-role coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8,192 | 0.839 ms | 0.782 | 0.812 | 0.801 | 96.8% |
| BF16 | 16,384 | 1.614 ms | 1.522 | 1.564 | 1.544 | 96.9% |
| BF16 | 32,768 | 3.211 ms | 3.038 | 3.114 | 3.070 | 97.0% |
| FP8 | 8,192 | 0.504 ms | 0.464 | 0.486 | 0.489 | 97.1% |
| FP8 | 16,384 | 0.966 ms | 0.900 | 0.932 | 0.939 | 97.2% |
| FP8 | 32,768 | 1.891 ms | 1.774 | 1.828 | 1.841 | 97.4% |

Attribution perturbs the 32K clean median by 4.6% BF16 and 1.3% FP8, so these
intervals localize work but are not acceptance timings.  They clear N2's 90%
coverage goal.  Intervals on different roles overlap and are not summed.

| Dtype | Rows/device | Source issue | Local issue | Writer Fabric issue | Writer credit | Receiver produced-wait | Receiver drain issue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8,192 | 0.211 | 0.202 | 0.280 | 0.204 | 0.133 | 0.655 |
| BF16 | 16,384 | 0.423 | 0.419 | 0.542 | 0.407 | 0.209 | 1.310 |
| BF16 | 32,768 | 0.843 | 0.812 | 1.105 | 0.836 | 0.383 | 2.636 |
| FP8 | 8,192 | 0.220 | 0.186 | 0.134 | 0.076 | 0.192 | 0.292 |
| FP8 | 16,384 | 0.437 | 0.385 | 0.253 | 0.152 | 0.531 | 0.404 |
| FP8 | 32,768 | 0.877 | 0.746 | 0.500 | 0.306 | 1.022 | 0.813 |

BF16's writer is narrowly critical, while its one-RISC drain overlaps almost
the entire writer interval.  FP8's dual-RISC receiver is narrowly critical and
is split between produced-wait and drain issue.  A second receiver core is not
justified: FP8 already uses both data-movement RISCs, and the prior BF16
dual-RISC experiment improved total latency by only 1.6%, below the 3% gate.
Proceed to N3 with producer/drain overlap as a joint pipeline problem.  Start
by changing sender prefetch depth without changing receiver slots or credits.

### Iteration N3: bounded producer/Fabric experiments

Run these one at a time in the accepted logical schedule.  Each gets a focused
test, release build, 20x reuse, exact counter check, and clean A/B/A before the
next begins.

| Order | Experiment | Metric expected to move | Keep gate |
| --- | --- | --- | --- |
| N3.1 | Separate source-read prefetch depth from receiver credit-window depth | Lower source idle/Fabric starvation without changing packet order | At least 3% combined median gain and stable command counts |
| N3.2 | Double-buffer sender L1/CB ownership at maximum legal payload rows | Overlap next source batch with current Fabric issue | At least 3% combined gain; no extra Fabric client or stale-slot failure |
| N3.3 | Partition existing sender work across its two data-movement RISCs while preserving one ordered Fabric producer | Lower source/local issue without bank-major payload permutation | At least 3% combined gain; route command order and credit counts unchanged |
| N3.4 | Move the existing sender/receiver core pair among legal CCL cores | Reduce NoC/Fabric contention without changing ownership or core count | At least 2% placement-only gain on both A/B/A controls |

**N3.1 result: rejected on 2026-07-18.**  The existing reader keeps two NOC
reads in flight in a three-entry CB.  An opt-in prototype separated that depth
from the receiver's 25/26-slot credit window and swept depths 1, 2, 3, 4, 6,
and 8.  All values passed BF16/FP8 isolated correctness.  The clean
`depth-2/depth-4/depth-2` seven-sample A/B/A result was:

| Dtype | Depth 2 A1 | Depth 4 | Depth 2 A2 | Depth 4 versus mean control |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.059 ms | 3.064 ms | 3.062 ms | 0.1% slower |
| FP8 E4M3 | 1.862 ms | 1.867 ms | 1.867 ms | 0.1% slower |

No preliminary depth improved either dtype by 1%, much less the 3% gate.
Increasing outstanding reads cannot reduce the measured per-page source issue
or Fabric/credit work, and two reads already hide source completion wait.  The
prototype and environment control were removed; production retains the
original two-read/three-entry pipeline.

**N3.2 result: rejected on 2026-07-18.**  An opt-in structural prototype
deferred each sender RISC's per-batch Fabric completion.  The reader could
issue the next source read while its previous payload remained in flight; the
writer retained its CB page until the following iteration, then completed and
released it before consuming the next page.  It did not add a core, Fabric
client, route, receiver slot, credit, or payload command.

The release build passed.  The candidate produced correct BF16 output across
three program-cache reuse invocations, and the combined batch-slice plus valid-
extent cases passed for BF16 and FP8 with realtime-profiler proof that the
receiver path was selected.  A same-process control/candidate cache test could
not allocate two receiver programs because each program's semaphores consume
the configured 512-byte L1-small region.  The test was narrowed to one
structural program rather than changing the production L1-small constraint.
This is a per-program test-resource limit, not a protocol failure.

The three-sample screening result at 32K rows/device was:

| Dtype | Automatic control median / p90 | N3.2 median / p90 | Median result |
| --- | ---: | ---: | ---: |
| BF16 | 3.061 / 3.065 ms | 3.066 / 3.076 ms | 0.2% slower |
| FP8 E4M3 | 1.861 / 1.861 ms | 1.869 / 1.876 ms | 0.4% slower |

The screen was far from the 3% keep gate, so the longer 20x and seven-sample
A/B/A acceptance ladder was not run.  A subsequent Fabric submission already
pre-flushes the preceding payload, so ownership deferral does not create a
deeper ordered Fabric window.  The prototype, test, and environment control
were removed.  Proceed to N3.3 with the accepted sender pipeline restored.

**N3.3 result: rejected by the measured upper bound on 2026-07-18.**  The
accepted sender already partitions work across its two data-movement RISCs:
NCRISC owns source reads and the forward ordered Fabric route, while BRISC
owns local-output writes and the backward ordered Fabric route.  N2 measured
their 32K critical intervals as 3.038 versus 3.114 ms for BF16 and 1.774 versus
1.828 ms for FP8.  Even an ideal zero-cost transfer of arbitrary work from the
busier RISC to the other can improve the maximum only until the two intervals
meet:

| Dtype | Reader / writer | Zero-cost balance limit | Maximum possible gain |
| --- | ---: | ---: | ---: |
| BF16 | 3.038 / 3.114 ms | 3.076 ms | 1.2% |
| FP8 E4M3 | 1.774 / 1.828 ms | 1.801 ms | 1.5% |

Both upper bounds are below the 3% keep gate before accounting for the cross-
RISC ownership semaphore, wait, and flush needed to keep CB reuse safe.  The
earlier dual-NoC sender prototype provides the empirical direction check: it
split source batches across these RISCs, passed the focused functional matrix,
and regressed BF16 by 3.1% and FP8 by 1.2%.  A new partition prototype cannot
meet N3.3's gate under the measured role balance, so no production code or
environment control was added.  Proceed to the placement-only N3.4 sweep.

**N3.4 result: accepted for FP8 on the reference 4x2 path on 2026-07-18.**
The isolated perf harness accepts an inclusive four-core rectangle through
`TTNN_AG_PERF_CORE_RECT`, implemented with the existing public sub-core-grid
argument.  Nine parser/validation cases pass.  Screening moved the unchanged
two sender/receiver pairs among horizontal and vertical logical-core ranges:

| Logical core rectangle | BF16 median | FP8 median | Result versus automatic screen |
| --- | ---: | ---: | --- |
| automatic horizontal | 3.069 ms | 1.864 ms | control |
| `(0,1)-(3,1)` | 3.067 ms | 1.866 ms | neutral |
| `(0,4)-(3,4)` | 3.045 ms | 1.871 ms | BF16 below 2%; FP8 regressed |
| `(0,7)-(3,7)` | 3.045 ms | 1.866 ms | BF16 below 2%; FP8 neutral |
| `(4,0)-(7,0)` | 3.229 ms | 1.860 ms | BF16 regressed |
| `(8,0)-(11,0)` | 3.348 ms | 1.834 ms | BF16 regressed; FP8 below 2% |
| `(0,0)-(0,3)` | 3.210 ms | 1.755 ms | FP8 wins; BF16 regresses |
| `(2,0)-(2,3)` | 3.239 ms | 1.745 ms | best FP8 screen, but requires a topology-specific offset |

The retained policy keeps BF16 and every explicit user sub-grid in the
established row-major order.  Automatic FP8 receiver plans without an explicit
sub-grid use deterministic column-major allocation, which selects the legal
`(0,0)-(0,3)` pairing without hard-coding a physical/core offset.  Kernel,
route, ownership, slot, credit, semaphore, payload, and Fabric-client counts
are unchanged.

Seven-sample `horizontal A1 / vertical B / horizontal A2` FP8 medians were
1.870 / 1.755 / 1.871 ms, with p90 1.880 / 1.761 / 1.879 ms.  The candidate is
6.2% faster than the mean control median and improves p90 by 6.3%.  After host
selection was changed, automatic production measurements were 3.063 ms BF16
and 1.753 ms FP8.  The release build, 9/9 selection/fallback matrix, 11/11
row-major matrix, and 40/40 combined BF16/FP8 reuse cases pass.  At that N3.4
checkpoint, the full Sparse MLA suite passed 39/39 with zero failures, skips,
or errors in 1003.296 seconds, including every scaled-FP8 4x2 short/long
DeepSeek and GLM case.  No dispatch timeout or tt-triage capture occurred.
The later final topology-policy candidate independently passes the same 39
cases in 950.50 seconds, as recorded above.

Do not add a receiver core in N3.  If all four experiments fail, the measured
Fabric/ERISC pipeline is the ceiling for this route and the next design study
may evaluate an explicit sender-side mux.  That study must reuse existing
Fabric endpoint clients, prove route ordering and resource capacity on the
host, and beat automatic by at least 5% before model integration.

### Iteration N4: product topology and core placement

For every target box/topology, enumerate legal core plans rather than copying
4x2 coordinates.  Reject a plan on the host if any sender/receiver overlaps a
router, dispatch, CCL, or model core, if ordinary L1 cannot hold the slot
window, or if L1-small cannot hold control semaphores.

Bring up in this order:

1. available QuietBox/loudbox linear configuration;
2. Galaxy linear control;
3. Galaxy ring on the production SP axis; and
4. Galaxy `FABRIC_2D_TORUS_Y` with its real direction partition.

For each rung, run one BF16 case, 20 repeats, FP8, the supported correctness
matrix, and only then perf.  Ring packets continue to target receiver Tensix
L1 through existing direction-specific sender routes.  Receiver cores never
open a Fabric connection.

### Iteration N5: model accounting

Run isolated CCL and model perf from the same build and configuration.  Report:

- native all-gather user-op count and device-program count per model step;
- main-MLA versus indexer AG call count and tensor bytes;
- direct and automatic AG median/p90 for each distinct shape;
- predicted model savings from AG time saved times call count;
- observed warm and long model latency; and
- unexplained residual, including overlap or synchronization evidence.

Do not compare a composite legacy operation count with a native device-program
count.  A lower time is a speedup; tables must state latency delta and speedup
with unambiguous signs.

### Phase 5: evaluate dual-RISC drain on one receiver core

**Status: implemented and measured behind
`TTNN_ALL_GATHER_RECEIVER_DRAIN_RISCS=2`.  Accepted as an FP8 candidate;
rejected as the BF16 default.**

**Prerequisite:** Attribution shows that receiver issue/drain occupies the
combined critical path.  Page reordering or placement in Phase 4 is not a
prerequisite because the parity prototype changes only RISC command-issue
parallelism and can be accepted or rejected independently.

**Hypothesis:** The receiver's two data-movement RISCs can raise local NOC
command issue throughput and reduce ingress/drain contention without adding a
Fabric connection or receiver core.

Tasks:

1. First partition complete batches by parity between BRISC and NCRISC.  This
   gives disjoint L1 slot offsets and output pages while changing only command
   issue parallelism.
2. Synchronize once per completed slot window, not once per batch.  RISC 0
   publishes `consumed(window)` only after both RISCs have flushed their
   assigned batches.
3. Keep initialization single-owner: RISC 0 resets produced state and releases
   RISC 1 through a local L1-small synchronization semaphore.
4. Run combined and drain-only modes only, with window credits.  Reject other
   combinations on the host before dispatch.
5. After the parity version is measured, partition by balanced bytes and NOC
   commands only if attribution still shows imbalance.
6. Assign page lists to the better NoC only as a separate follow-up change.
7. Test odd batches, one-page tails, partial extents, a final partial window,
   and an empty second-RISC assignment.

The first implementation deliberately uses BRISC and NCRISC with their
default distinct NoCs and leaves Fabric routes unchanged.  This isolates local
receiver command parallelism.  It must not be described as additional Fabric
fan-out.

Measured exit result:

- focused BF16 and FP8 profiler-path tests: pass;
- sliced-partial reuse: 20/20 per dtype;
- full row-major matrix: 11/11;
- drain-only: 34.9% faster BF16 and 34.7% faster FP8;
- combined: 1.6% faster BF16 and 15.1% faster FP8; and
- p90: improved for both dtypes in the clean A/B/A session.

Keep dual-RISC for FP8 experimentation.  Retain the simpler single-RISC BF16
default until an upstream change makes receiver drain critical again.

### Phase 6: evaluate receiver-core fan-out

**Prerequisite:** One receiver remains issue- or drain-limited after Phase 5.

**Hypothesis:** Two carefully placed receiver cores can raise destination DRAM
write bandwidth while using the same number of Fabric client connections.

Tasks:

1. Allocate two legal receiver cores per active link.
2. Give them disjoint output page sets and disjoint L1 slot regions.
3. Encode the selected receiver core's L1 NOC address in each Fabric packet.
4. Stripe complete transport batches first; consider per-bank striping only if
   it reduces imbalance without splitting a packet.
5. Aggregate local consumption on the existing sender core.
6. Return exactly one remote credit per destination/window through the
   existing sender routes.
7. Measure receiver imbalance, Fabric ingress bandwidth, DRAM bandwidth, and
   sender stall time.

No receiver core opens a Fabric connection.  If two targets require more
outstanding Fabric state than the existing route supports, use scheduling or
multiplexing on the existing sender route rather than silently acquiring a new
ERISC client channel.

Exit: keep fan-out only if G8 passes.  If the raw-ingress rate is unchanged and
the second receiver is idle, stop; the sender/Fabric front end is limiting.

### Phase 7: generalize dispatch and fallback

**Hypothesis:** Host-side page-plan validation can select the receiver only for
safe mappings with negligible dispatch overhead.

Tasks:

1. Replace MLA-shape predicates with page-plan/resource predicates.
2. Validate packet size, L1 use, number of streams, core plan, and route plan.
3. Add explicit rejection reasons for unsupported mappings.
4. Include the receiver policy in program identity or remove mutable
   environment dependence from production dispatch.
5. Retain `force_direct` and safe `force_receiver` controls for tests only.
6. Run generic row-major, tiled, sharded, partial, sliced, and persistent-output
   fallback tests.

Exit: G9 passes.  Unsupported cases never enter a partially valid receiver
program and then fail on device.

### Phase 8: product topology validation

Do not infer Galaxy behavior from the 4x2 linear proxy.

For each available QuietBox/loudbox and Galaxy setup, record:

- device count and physical mesh;
- active mesh axes;
- linear, ring, or `FABRIC_2D_TORUS_Y` routing;
- number of links and sender/receiver cores per link;
- receiver core coordinates and bank ownership;
- BF16 and FP8 correctness/stability;
- direct versus receiver median/minimum/p90; and
- whether fallback was selected and why.

Ring is a supported production topology.  The receiver design must use the
existing ring routes and their direction partition; it must not assume a
single linear predecessor/successor or add an endpoint client.

Exit: G10 passes for every available required deployment.  A topology-specific
failure stays disabled by a host predicate until fixed.

### Phase 9: Sparse MLA integration and end-to-end measurement

Only start model sign-off after native partial/slice receiver support passes.

Tasks:

1. Run `test_sparse_mla.py` correctness with the ordinary model API.
2. Assert that intended main-MLA and indexer gathers select receiver mode.
3. Record the number of collective/device operations before and after; do not
   count a formerly composite all-gather as multiple independent user ops.
4. Measure isolated all-gather time inside the model shape.
5. Measure `test_sparse_mla_perf.py` end-to-end time separately.
6. Report expected model savings as measured AG time saved multiplied by AG
   call count, then compare it with observed end-to-end savings.
7. Explain any gap using profiler overlap, synchronization, or other model
   stages rather than mixing composite-op and device-kernel timing.

Exit: G11 passes.  The final report contains both isolated AG and model timing,
with no table that labels a latency increase as a speedup.

**Available P150 status: complete on 2026-07-18.**  The model now supplies a
persistent interleaved DRAM output for the main KV-prefix gather and the
receiver accepts its ND-sharded DRAM source when page geometry matches exactly.
The output-return path preserves the cache-owned Python tensor wrapper; this
fixed a red cold-reuse test where cleanup through the temporary operation-return
alias deallocated the persistent allocation before the second forward.

The full Sparse MLA suite passes 39/39 after the integration.  A receiver-
confirmed SP=4 x TP=2 A/B/A shows 10.34-13.76% BF16 CCL wins and 36.51-42.63%
FP8 CCL wins.  End-to-end gains are 0.35-0.47% BF16 and 4.40-5.58% FP8 at the
warm cache, and 3.99-5.11% BF16 and 12.13-15.69% FP8 at the long cache.  Every
automatic warm/long run contains one receiver program and every forced-direct
run contains zero.  Direct and automatic program counts are identical, proving
that the result changes one collective implementation rather than adding a
model op.  Actual QuietBox and Galaxy topology sign-off remains under G10/I11.

The final available-P150 closure rerun also passed 25/25 focused receiver,
selection, resource, and fallback cases, followed by 60/60 repeated BF16,
BF16 one-page-tail, and FP8 geometry cases.  All invocations used the safe
runner and release firmware; neither closure run hung or required triage.

## Correctness matrix

| Dimension | Required coverage |
| --- | --- |
| Data type | BF16, FP8 E4M3 |
| Page geometry | 1152 B MLA BF16, aligned 704 B FP8, one-page tail, ordinary non-MLA pages |
| Gather selection | full, `batch_slice_idx`, `valid_gather_extent`, combined slice+extent |
| Output | fresh output for generic coverage; persistent/preallocated output for all perf and MLA cases |
| Mapping | receiver-eligible row major, non-contiguous fallback, tiled fallback, sharded fallback where supported |
| Slots/windows | 1, 2, automatic maximum, final partial window |
| Program reuse | 20 consecutive invocations with changing input and output addresses where supported |
| Topology | 4x2 linear, required QuietBox layouts, Galaxy linear/ring/Torus-Y |
| Receiver plan | one core/one NoC; one core/two NoCs; two cores only when those variants exist |

Every receiver test asserts:

- numerical output;
- selected path by realtime-profiler kernel source or equivalent test metadata;
- persistent output identity when requested;
- untouched sentinel regions for partial output;
- no L1 allocation overlap; and
- exact page-owner coverage.

## Performance methodology

### Timing rules

1. Use the realtime device profiler for isolated collective timing.
2. Discard the first profiled invocation because delayed warmup records can be
   attributed to it.
3. Use at least seven timed samples for isolated sweeps.
4. Report raw samples, median, minimum, and p90.
5. Run A/B/A, with direct mode before and after the candidate.
6. Repeat the session if the two direct medians differ by more than 3%.
7. Use identical tensors, persistent output, Fabric routing, warmup count, and
   profiler configuration for both paths.
8. Report physical bytes, logical bytes, packets, windows, and effective
   payload bandwidth.
9. Time model end-to-end performance separately from isolated AG.

### Required comparison rows

| Comparison | Purpose |
| --- | --- |
| direct remote DRAM | production baseline |
| automatic receiver | current candidate floor |
| forced receiver with one slot/per-slot credit | protocol-cost diagnostic |
| safe L1 sink | ingress plus credit cost |
| L1 overwrite | unsafe raw-ingress upper bound only |
| drain-only | receiver NOC/DRAM cost |
| one receiver, bank-aware | page-schedule value |
| one receiver, dual-NoC | intra-core drain parallelism |
| two receivers | fan-out value |

Diagnostic modes are not production candidates and must be clearly labeled as
such in every table.

### Result record

Each measured commit appends a row or artifact containing:

| Field | Required value |
| --- | --- |
| Hypothesis | one expected bottleneck and expected metric movement |
| Commit | exact candidate and baseline commit IDs |
| Hardware | architecture, box, device count, mesh |
| Routing | topology and Fabric configuration |
| Tensor | shape, dtype, logical row bytes, physical page bytes |
| Selection | full/slice/valid extent and gathered dimension |
| Buffers | persistent output, L1 base/end, slots and bytes |
| Control | semaphore addresses/memory class, credit mode, notify mode |
| Core plan | sender and receiver coordinates, NoC/page ownership |
| Attribution | enabled/disabled, marker version, critical reader/writer/receiver intervals |
| Samples | warmups, raw samples, median, minimum, p90 |
| Throughput | payload bytes and effective GB/s |
| Correctness | selector and repetition count |
| Diagnostics | safe-runner result and tt-triage artifact if any |
| Decision | keep, revise, disable, or revert, with threshold |

## Commands

All pytest invocations must use the safe runner.  Never invoke pytest directly
for this work and never pass `--dev`, because it enables watcher/LLK assertions
and changes firmware code size.

### Release build

```bash
./build_metal.sh --release
```

### Focused CCL correctness

```bash
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_selection_paths' -q -s

scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'test_all_gather_fabric_2d_row_major_2k_pages' -q -s
```

### Realtime-profiler path assertion

```bash
TT_METAL_DEVICE_PROFILER=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_selection_paths' -q -s
```

### Stability

```bash
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_selection_paths and batch_slice_partial_extent_receiver' \
  --count=20 -q -s
```

The selector covers both BF16 and FP8 combined selection cases.

### Isolated direct baseline

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_direct \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s
```

### Isolated automatic receiver

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=auto \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s
```

The result must record the effective values of:

- `TTNN_ALL_GATHER_RECEIVER_STAGE_MODE`;
- `TTNN_ALL_GATHER_RECEIVER_SLOTS`;
- `TTNN_ALL_GATHER_RECEIVER_BATCH_ROWS`;
- `TTNN_ALL_GATHER_RECEIVER_NOTIFY_MODE`; and
- `TTNN_ALL_GATHER_RECEIVER_CREDIT_MODE`.

### Bank-owned correctness and A/B/A gate

The experimental schedule switch is `TTNN_ALL_GATHER_BANK_OWNED_LINKS=1`.
`TTNN_ALL_GATHER_BANK_OWNED_COALESCE` independently selects `none`, `source`,
`source_local`, or `all`; all four stages are implemented.
`TTNN_ALL_GATHER_BANK_OWNED_RUN_POLICY=divisor|max_tail` selects R8/R16 or
R12/R20 in the host factory and is printed by the perf test.  These controls
are diagnostic, not production APIs.  The completed experiment can be
reproduced by first running focused correctness:

```bash
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_BANK_OWNED_LINKS=1 \
TTNN_ALL_GATHER_BANK_OWNED_COALESCE=source_local \
TTNN_ALL_GATHER_BANK_OWNED_RUN_POLICY=max_tail \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'test_all_gather_fabric_2d_bank_owned_schedule' -q -s
```

Then run each timing leg in a separate process so the program cache cannot
reuse a differently compiled policy:

```bash
# A: exact-divisor bank-owned control
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_BANK_OWNED_LINKS=1 \
TTNN_ALL_GATHER_BANK_OWNED_COALESCE=source_local \
TTNN_ALL_GATHER_BANK_OWNED_RUN_POLICY=divisor \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s

# B: maximum-run tail candidate; every other setting is identical
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_BANK_OWNED_LINKS=1 \
TTNN_ALL_GATHER_BANK_OWNED_COALESCE=source_local \
TTNN_ALL_GATHER_BANK_OWNED_RUN_POLICY=max_tail \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s

# A again: repeat the exact-divisor command unchanged.
```

Before timing B, run its focused correctness selector once and then 20 times.
With `TTNN_ALL_GATHER_RECEIVER_ATTRIBUTION=1`, verify the exact source, local,
Fabric-payload, receiver-write, and credit counts.  Disable attribution for the
acceptance timing.  Discard the session if the two A medians differ by more
than 3%.  If `max_tail/source_local` wins, hold `max_tail` fixed and run a
second `source_local/all/source_local` A/B/A to determine whether larger
payloads move receiver coalescing back onto the critical path.  Finally compare
the best static mode against the accepted automatic receiver in
automatic/static/automatic order; the production keep gate remains 3%.

The recorded outcome is: max-tail wins inside the static path, `all` loses to
`source_local`, and max-tail/source-local loses to automatic.  These commands
are retained for reproduction and attribution, not as the next optimization
queue.

### Drain-only attribution

This is a diagnostic mode.  It writes unspecified L1 contents to the
persistent output and therefore measures timing only; it must not be used for
output correctness.

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_RECEIVER_STAGE_MODE=drain_only \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s
```

### Per-RISC cycle attribution

Attribution requires the realtime profiler.  Run the test to completion first;
the device CSV is materialized when the mesh closes.  Analyze it afterward in
a separate command:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_RECEIVER_STAGE_MODE=combined \
TTNN_ALL_GATHER_RECEIVER_ATTRIBUTION=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s

scripts/analyze_all_gather_attribution.py --samples 7
```

The test prints `AG_ATTRIBUTION_READY` with normalized runtime IDs.  Pass
those IDs through `--runtime-ids` when the profiler CSV contains unrelated AG
runs.  Attribution records from reader, writer, and receiver roles overlap;
compare their critical paths but do not sum them.

### Active dual-RISC receiver experiment

The first gate JIT-compiles both receiver kernels and checks the focused BF16
combined path:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_ALL_GATHER_RECEIVER_DRAIN_RISCS=2 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_selection_paths and bf16_batch_slice_partial_extent_receiver' \
  -q -s
```

If that passes, run the corresponding FP8 selector and then the combined
BF16/FP8 selector 20 times.  Do not begin timing after a one-off pass.

Clean combined timing, with attribution disabled, is:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TTNN_RUN_AG_ISOLATED_PERF=1 \
TTNN_AG_PERF_ROWS_PER_DEVICE=32768 \
TTNN_AG_PERF_SAMPLES=7 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_receiver \
TTNN_ALL_GATHER_RECEIVER_STAGE_MODE=combined \
TTNN_ALL_GATHER_RECEIVER_DRAIN_RISCS=2 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'sparse_mla_row_perf' -q -s
```

Repeat with `TTNN_ALL_GATHER_RECEIVER_DRAIN_RISCS=1` in a separate process for
the control, and use A/B/A ordering.  Then repeat both RISC counts with
`TTNN_ALL_GATHER_RECEIVER_STAGE_MODE=drain_only`.  Only after clean timing may
attribution be enabled to explain the result.

### Sparse MLA functional and performance

```bash
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -q -s

scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py -q -s
```

### Hang handling

If a test hangs:

1. let `scripts/run_safe_pytest.sh` collect tt-triage and reset safely;
2. preserve `generated/tt-triage/triage.csv` and related artifacts;
3. identify every stalled core, kernel, source line, semaphore address/value,
   expected sequence, and Fabric connection state;
4. reduce the failure to one isolated all-gather case;
5. fix protocol progress before resuming performance sweeps; and
6. do not increase timeouts or use `--dev` as a workaround.

## Immediate experiment queue

| Order | Status | Standalone change | Test-first evidence | Keep criterion |
| --- | --- | --- | --- | --- |
| I0 | Done | Validate selection dispatch | Six receiver selections plus one fresh-output fallback | 7/7 normal and profiler-path cases pass |
| I1 | Done | Native `batch_slice_idx` receiver plan | Slice test forced to receiver, persistent output | BF16/FP8 correct and receiver selected |
| I2 | Done | Native `valid_gather_extent` receiver plan | Extent test forced to receiver | BF16/FP8 correct and receiver selected |
| I3 | Done | Combined slice+extent | Combined focused test | BF16 and FP8 each pass 20 consecutive runs |
| I4 | Done | Drain-only calibration mode | Exact production page plan, timing-only output | 1.404 ms BF16 and 1.270 ms FP8 |
| I5 | Done | Add source/Fabric/wait/credit/barrier attribution | BF16/FP8 combined and drain-only role breakdown | Receiver per-page drain issue and ingress contention identified; BF16 combined overhead about 0.5% |
| I6 | Done | Dual-RISC one-core drain, parity partition | 40/40 reuse plus 11/11 row-major coverage | Accepted for FP8 at 15.1%; rejected as BF16 default at 1.6% |
| I7 | Done | Sweep bytes, packets, and windows independently | Three byte sizes, four packet sizes, and four window sizes per dtype | Shared ~37 GB/s byte ceiling; maximum legal packet/window is best |
| I8a | Done | Pure two-link/eight-bank host schedule | 3/3 reference cases pass: 32/R4, 32768/R12, 32768/R20 | Exact full/tail coverage and identical schedule at source, local writer, and receiver |
| I8b | Done | Schedule-only bank ownership with per-page NOC commands | Focused BF16/FP8 pass; repeated run passes 40/40 through the safe runner | Correct, stable, and no protocol hang |
| I8c | Done | One contiguous source read per bank run | Exact 8x/16x source-count reduction; focused correctness and 40/40 stability pass | Retained experimentally: 0.5-0.6% clean incremental win |
| I8d | Done | One contiguous local-output write per bank run | Exact local count reduction and focused correctness | Retained experimentally: another 0.6% BF16 / 1.1% FP8 win |
| I8e | Done, rejected for selection | One contiguous receiver write per bank run | Exact 8x/16x receiver count reduction and focused correctness | 0.4-0.6% latency regression; receiver is off critical path |
| I8f | Done, rejected for production | Maximum R12/R20 runs with final per-bank tails | Pure tail cases, focused BF16/FP8, exact counts, and clean A/B/A pass | Max-tail beats divisor, but best static remains 4.1% slower BF16 / 1.7% slower FP8 than automatic |
| I9 | Skipped for static path | Offset/link-count generalization and legal core-pair sweep | Static path failed its bounded production gate | Reopen placement only for an accepted path with measured NoC contention |
| I10 | Not justified | Two receiver cores/link | Current receiver waits for production | Reopen only if attribution proves receiver-side headroom; require 5% combined gain and no extra Fabric connection |
| I11 | Partial | Product routing enablement for automatic receiver | Supported P150 8x1 ring path-selection and A/B are complete; run the topology-specific ladder on QuietBox and actual Galaxy | Correct/stable with separate direct/automatic perf record for every deployed route |
| I12 | Done on available P150 | Sparse MLA sign-off | 39/39 post-integration correctness, cold 11-forward reuse per dtype, and receiver-confirmed SP=4 warm/long A/B/A recorded | Actual QuietBox/Galaxy end-to-end sign-off remains coupled to I11 product-topology access |

Stop an iteration immediately on a hang, output mismatch, duplicate page owner,
L1 overlap, unexpected fallback, route/client increase, command-count mismatch,
or release-build failure.  Do not start receiver-core fan-out unless the
link-owned one-core-per-link experiment shows receiver-side headroom.  Do not
keep a command-count reduction merely because an isolated interval improves;
combined latency and p90 own the production decision.

### Static bank-owned execution ledger

| Date | Candidate | Release/JIT gate | Correctness/stability | Performance decision |
| --- | --- | --- | --- | --- |
| 2026-07-18 | Pure reference schedule | Python collection/import succeeds | 3/3 schedule proofs pass | Not timed; host-only proof |
| 2026-07-18 | Schedule-only source/local/receiver enumeration | Release build passes; device kernels JIT | BF16 and FP8 focused cases pass; 40/40 repeated cases pass | Not yet accepted on latency; serves as the coalescing control |
| 2026-07-18 | Source-read coalescing only | Release build passes; initial JIT narrowing warning fixed with explicit NOC-coordinate casts | Focused BF16/FP8 and 40/40 repeated cases pass | Exact 8x/16x source reduction; clean incremental win 0.5-0.6% |
| 2026-07-18 | Source + local-output coalescing | Release build and focused cases pass | Source and local counts are 2,048 BF16 / 1,024 FP8 per worker | Clean incremental win 0.6% BF16 / 1.1% FP8 |
| 2026-07-18 | Source + local + receiver coalescing | Release build and focused cases pass | Receiver counts fall 49,152 to 6,144 BF16 and 24,576 to 1,536 per critical FP8 RISC | Rejected for selection: total latency regresses 0.4-0.6% |
| 2026-07-18 | Tail-aware maximum R12/R20 runs | Release build passes after removing obsolete divisor assertion; focused BF16/FP8 and 40/40 stability pass | Source/local/Fabric counts exactly 1,368 BF16 / 820 FP8; BF16 receiver aggregate 4,104 | Max-tail is 14.4% / 8.2% faster than divisor, but best static is 4.1% / 1.7% slower than automatic; diagnostic-only |

No row above required hang triage.  If a later row times out, preserve the
safe-runner artifact and diagnose it with tt-triage before modifying batching,
timeouts, or firmware configuration.

### Dual-RISC decision record

The dual-RISC experiment was executed in this order; steps 1 through 6 are
complete and step 7 selected source/Fabric scaling as the next work:

1. **Compile/correctness gate.** Run the focused BF16 dual-RISC selector once.
   If compilation fails, repair only the dual-RISC kernel/runtime contract.  If
   it hangs, use the safe-runner tt-triage artifact and inspect the local
   produced, dual-sync, consumed, and completion sequences before changing any
   performance parameter.
2. **Semantic gate.** Run FP8, odd/final batches, sliced partial extent, and
   sentinel checks.  The two RISCs must have disjoint output ownership and
   RISC 0 must publish consumption only after RISC 1 completes the window.
3. **Reuse gate.** Repeat the combined BF16 and FP8 selection cases 20 times.
   Any stale sequence or intermittent timeout rejects the protocol even if a
   single invocation is fast.
4. **Clean drain A/B/A.** Compare one versus two RISCs in drain-only mode with
   attribution disabled.  This answers whether local command issue scaled.
5. **Clean combined A/B/A.** Compare one versus two RISCs in combined mode.
   Keep only a repeatable median improvement of at least 3% with no p90
   regression beyond 3%.
6. **Explain the result.** Enable attribution only after clean timing.  A
   successful result should reduce receiver drain-issue time and ideally the
   gap between drain-only and combined.  If it does neither, remove or disable
   the dual-RISC production path.
7. **Choose the next single variable.** If two RISCs help but remain
   imbalanced, try bank/NOC-aware ownership.  If receiver time falls but total
   time does not, return to source/Fabric packet and credit work.  Add a second
   receiver core only if receiver drain remains the measured combined-path
   limiter.

The expected outcomes and next actions are:

| Observation | Interpretation | Next action |
| --- | --- | --- |
| Drain-only and combined both improve | Receiver issue was on the critical path | Keep dual-RISC experimentally; run full sweeps and topology gates |
| Drain-only improves, combined is flat | Sender/Fabric or shared-NoC contention masks the drain gain | Do not enable by default; profile overlap and sender/Fabric path |
| Combined improves only for BF16 | Byte-dependent receiver pressure remains | Keep dtype/shape predicate only if generic and stable; fit byte/packet model |
| Combined improves only for FP8 | Fixed command rate is dominant | Check command balance and synchronization overhead before generalizing |
| Neither improves | One core is not command-issue limited in practice, or synchronization cancels the gain | Revert/disable dual-RISC and move to address-run/static-ownership analysis |
| Hang or stale output | Cross-RISC epoch/ownership protocol is invalid | Stop perf work; minimize and diagnose with tt-triage |
| Two direct-control medians drift by more than 3% | Measurement session is unstable | Discard the session and rerun A/B/A |

## Commit strategy

Keep reviewable standalone commits in this order:

1. protocol and completion correctness;
2. test observability and isolated perf harness;
3. automatic slots/window policy;
4. sliced gather receiver support;
5. partial gather receiver support;
6. drain-only instrumentation;
7. opt-in per-RISC attribution and analysis tool;
8. dual-RISC receiver drain, if accepted;
9. physical-address attribution and rejected-permutation decision record;
10. pure bank-owned host schedule and ownership tests;
11. schedule-only two-link kernel path;
12. source-run coalescing, if accepted;
13. local-output-run coalescing, if accepted;
14. receiver-run coalescing, if accepted;
15. core placement or receiver fan-out, each only if independently accepted;
16. product-topology support; and
17. model integration/performance record.

Rejected performance experiments may be retained as diagnostic commits on an
experiment branch, but rejected modes must not remain selected by production
dispatch.  Each accepted commit includes its focused test and updates the
result ledger.

## Definition of done

The new all-gather work is complete only when all of the following are true:

1. Release build passes without `--dev`.
2. Full, sliced, partial, and sliced-partial BF16/FP8 gathers are correct with
   persistent output.
3. Output payloads use ordinary L1 staging and control semaphores use L1-small.
4. Program reuse is stable for at least 20 consecutive invocations per focused
   dtype/selection class.
5. Unsupported cases select the existing generic path before dispatch.
6. The selected production receiver configuration beats direct AG on the
   isolated reference workloads with stable median and p90.
7. Every extra receiver RISC, NoC policy, or receiver core retained in
   production has a separately measured incremental benefit.
8. QuietBox and Galaxy supported routing modes are correct and measured.
9. Sparse MLA main and indexer gathers select the intended implementation.
10. Sparse MLA functional tests pass and its perf report separates isolated AG
    latency from end-to-end model latency.
11. The final result table contains exact commits, commands, hardware, raw
    samples, and an unambiguous statement that lower latency is faster.

## Non-goals

- Repacking or padding the MLA KV cache solely to improve all-gather.
- Replacing the generic scatter fallback for mappings not covered by a proved
  receiver page plan.
- Adding receiver Fabric connections to obtain more endpoint channels.
- Adding cores before profiling proves that the current receiver is limiting.
- Using `--dev`, larger timeouts, or firmware-size changes to hide a progress
  bug.
- Comparing composite operation timing with a single device kernel and calling
  the difference an all-gather speedup.
- Claiming Galaxy or ring support from a linear 4x2 proxy result.
