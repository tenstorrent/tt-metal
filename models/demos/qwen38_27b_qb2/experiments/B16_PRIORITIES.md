# B16 per-user performance priorities

Updated October 10, 2026 UTC. Primary target: one TP4 replica at batch 16,
32K total context first, 16K second. Retain 128K/256K support at feasible
concurrency. B32 is a secondary regression screen. Keep BFP8 weights/KV and
FP32 recurrent state. Prefill and decode share the Galaxy.

## Measured baseline and proposed gains

The completed shared-QK full-model sweep measured 14.871 decode tokens/s/user
at 32K/B16 and 16.337 at 16K/B16. For 32K input and 128 output tokens, one TP4
measured 5,326 prefill input tokens/s, 98.59-second median TTFT and 19.12
aggregate output tokens/s including prefill. These are model-harness results,
not an HTTP serving benchmark. Eightfold scaling is not implied.

| Change | Expected B16 benefit | Evidence and qualification |
|---|---|---|
| Prefill token budget 32K to 64K | About 10% input throughput; roughly 99 to 90 seconds TTFT at 32K/B16; about 9% all-in output throughput for 128 output tokens | Earlier budget experiment measured 5,321 to 5,870 input tokens/s at B16. B32 failed allocation. Queue a fresh matched B16-only control/candidate/control comparison. Changed chunk boundaries require accuracy requalification. |
| Direct preparation plus fused GDN epilogue | Projected 14.87 to 16.54 decode tokens/s/user at 32K, about 11%; 16.34 to 18.38 at 16K, about 12.5% | Real-layer comparison saves 6.80 ms over 48 layers. Full-model comparison and GPQA are running. About 1% all-in benefit for the prefill-dominated 32K/128-output batch. |
| Broader compact GDN front end | Engineering target: another 10-20% decode throughput on the combined-fusion candidate | Not measured. Targets convolution, history, preparation and projection-layout work. The first prototype implements only convolution/history and compact preparation; do not assign it the entire broader-front-end gain. |
| Scheduler chunked prefill | Reduce decode pauses under arriving prompts; no raw compute-throughput gain assumed | Workload-dependent benefit is unquantified. Older BFP4 adapter state test passed. Current BFP8 scheduler, device sampler and mixed-load serving tests are still needed. |

For each further optimization, record baseline, workload, per-op change,
estimated full-step saving, expected TTFT/TSU/all-in improvement, uncertainty,
test duration and acceptance evidence **before** running it. Then replace the
estimate with measured results, retaining regressions and failed attempts.
Do not add percentage speedups or count already-selected buffering twice.

## What the new profiles show

All four 32K B16/B32 control/candidate Tracy captures completed. They are warm
eager profiles of real-weight layers 0 and 3 with synthetic populated caches.
They do not constitute full-model traced critical-path reconciliation.

At B16, the combined-fusion candidate has the following median kernel sums
across the four TP ranks. Inclusive and exclusive regions must not be mixed.

| Region | Time per representative layer | Model multiplicity |
|---|---:|---:|
| Packed convolution, inclusive | 110.46 us | 48 |
| Recurrence and input preparation, inclusive | 115.89 us | 48 |
| GDN packed projection and its layouts, inclusive | 124.83 us | 48 |
| Other work directly inside GDN, exclusive | 130.95 us | 48 |
| Paged attention and surrounding ops, inclusive | 796.77 us | 16 |
| MLP gate/up projection, inclusive | About 120 us | 64 |
| MLP down projection, inclusive | About 91 us | 64 |

At B16, three output tilizations account for about 73 us of the convolution
region, roughly 3.5 ms over 48 layers. The compact prototype removes the
32-time-row-per-user output representation and computes only one retained
convolution output per user. It keeps the native four-tap operation order,
BF16 partial packs and precise SiLU. It also reads old history before shifting
each disjoint channel range in place.

Useful traffic at 32K/B16 is approximately 7.113 GB weights, 4.563 GB KV reads
and 1.208 GB recurrent-state read/write per chip and step. The completed
14.87-TSU model implies 191.6 GB/s of useful traffic, or 37.4% of an assumed
512 GB/s peak. This is a model, not a DRAM utilization counter. The attention
kernel alone is around 70% of its useful-KV bandwidth bound. NoC congestion
has not been established as the dominant cause. Extra traffic, dependencies,
layout conversions, compute and synchronization contribute to the gap.

### Comparison with Blaze reload

The inspected Blaze reload design at `0ecfc5099203387554a5ca2912f066c098a45fd0`
records 24.75 MB / 52.5 us = 471 GB/s, or 92% of the assumed peak, for expert
streaming matmuls with BFP4 weights. Its sections 4.7 and C.1 explicitly say
this is not measured whole-model reload efficiency. Streaming kernels place
one worker adjacent to each bank, read contiguous bank-local shards, aggregate
packets, and keep tagged reads in flight while compute consumes earlier data.
The reload planner also balances bytes and modeled NoC links across ports and
planes. Those mechanisms are relevant without changing Qwen precision.

Our BFP8 read-only probe already reached 499-508 GB/s. Adding remote consumers
in the completed 54-variant delivery sweep reduced the best useful rate to
275-280 GB/s, before attention math or production page-table traversal. This
establishes a cost in that delivery/receiver design, not a diagnosis of hardware
NoC saturation. Its output-byte accounting and the full-model useful-byte
estimate have different denominators from the expert-matmul measurement.

The current attention path is faster than that delivered-reader prototype,
so installing it would not be a justified optimization. Prioritize compact
GDN data flow and fewer intermediate layouts now; a future bank-local reader
must include consumer delivery, backpressure and compute in its acceptance
measurement. Reader proximity or deeper queues alone are not new gains: the
probe already showed packet aggregation and placement must work together,
and depths four/eight are almost equal in the best delivery configuration.

Evidence: [read calibration](../galaxy-evidence/shared-qk-and-bandwidth-v1/README.md)
and [completed delivery sweep](../galaxy-evidence/delivery-extended-v2/README.md).

At the user's request, reconstructed the control's approximate model cost from
the current two-layer profile: multiply layer-0 exclusive operations by 48,
layer-3 operations by 16, count outer model operations once, then take medians
across ranks. This extrapolates representative eager kernels; it is not a
full-model traced critical-path measurement and does not complete the P0 gate.

| Operation family | Extrapolated time per B16/32K step |
|---|---:|
| Matmuls | 18.69 ms |
| SDPA | 12.59 ms |
| Layout, padding, slicing and conversions | 19.49 ms |
| Remaining kernels, including recurrence, norms and collectives | 13.60 ms |

The layout family includes tilize, untilize, slice, reshape, fill-pad, reshard,
typecast, concat, transpose and interleaved/sharded conversion operations.
Their entire runtime is not necessarily removable. Family medians sum to
64.37 ms versus measured traced model TPOT 67.25 ms; that numerical agreement
does not establish identical traces, activations or per-layer timings.
Dividing modeled weight bytes by matmul time suggests 380 GB/s (74% of peak),
and modeled KV bytes by SDPA time suggests 363 GB/s (71%). These are estimates,
not physical traffic counters. They support removing intermediate operations
as a major opportunity rather than diagnosing every reader as running at 37%.
At 90% of peak, useful traffic alone takes 27.96 ms; the roughly 39-ms gap to
the measured step includes necessary compute, extra traffic and overhead.
It is not a promise of 39 ms recoverable time.

Source: [control profile](../galaxy-evidence/gdn-fusion-progress-v3/gdn-fusion-full-v2/profile-s32768-b16-native/analysis/profile-summary.json).

## Queued work and scope

`qwen38-b16-priority-v2-20261010.service` waits for the exact invocation of
the existing fusion/GPQA queue. It then takes the common device lock for:

1. B16 full-model sweeps at 32K and 16K, with 32K/64K/32K prefill budgets.
   All three use the shared-QK BFP8 policy; only the budget changes.
2. Nine existing direct-preparation hardware regression cases, since the
   experimental compact-input reader shares the implementation.
3. Eight compact convolution/preparation cases, B16 first, then B32, with
   L1/DRAM and public/compact projection inputs. Every TP rank must match
   native convolution and prepared values bit-for-bit, including changed-input
   trace replay, separate allocations, and in-place history updates.

The compact kernel is not selected by a model policy or serving default.
Full GDN-layer integration, full-model performance, and GPQA follow any passing
standalone screen. The broader 10-20% target remains unproven.

The follow-up survives SSH/client disconnects, not a host reboot. Source hashes,
timeouts, memory limits, individual logs and receipts are retained. The first
staging attempt failed CPU preflight because a test fixture required an error
message argument; it opened no hardware. The corrected preflight passed
491 tests and 40 subtests, with one unrelated skip.

Serving is currently 8 TP4 workers with a 1/8/16 decode bucket set. Configured
per-request context is 262,144 input-plus-output tokens. The configured KV pool
is shared per TP4, so maximum context and maximum concurrency are not jointly
guaranteed. Prefix caching and scheduler chunked prefill remain disabled.

Evidence: [completed benchmark and GPQA](../galaxy-evidence/perf-priority-completed-v1/README.md),
[four current profiles](../galaxy-evidence/gdn-fusion-progress-v3/capture.json),
and the [timeline](LOGBOOK.md).
