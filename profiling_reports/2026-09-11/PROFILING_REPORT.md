# Mistral Small 4 prefill profiling - September 10 results

## Outcome

Both performance baselines completed successfully on `bh-glx-120-b10u14`, a 32-device Blackhole Galaxy. TP's full operation profile passed validation. PP4's capture completed, but its final rank's replay attribution failed validation; the complete PP4 operation breakdown remains unfinished. No performance optimization was implemented or claimed.

All runs used `akhan/mistral4-prefill-followups` at `b48bf4095de1601786c4cf0e91a647a532c730c5`.

**Evidence retention caveat:** this report reconstructs the results recorded in the completed September 10 session. On September 11, the old `/tmp/ssalice-mistral4-prefill-followups-profile` worktree and its artifacts were absent, confirmed both inside the sandbox and on the host. The logs and CSVs from those runs cannot currently be rechecked. This report is not a replacement for the lost primary artifacts. Historical September 8 captures used for the separate B1 investigation still exist and are independently checked there.

## Matched baseline

36 layers, 5,120-token requests, two users, maximum sequence length 10,240, traced execution, `GPT_DEVICE` routing, KV-only final layer. Each baseline processed 48 one-chunk requests. TP used SP8/TP4 on all 32 devices; PP4 used four SP8/TP1 stages, nine layers each.

| Configuration | Mean cadence | Median | p90 | Derived input tokens/s |
| --- | ---: | ---: | ---: | ---: |
| TP | 127.604 ms | 127.577 ms | 127.881 ms | 40,124 |
| PP4, final rank | 97.006 ms | 97.008 ms | 97.206 ms | 52,780 |

The measured window was `CHUNK_START` c8 through c47: 39 start-to-start intervals. Throughput is 5,120 divided by mean cadence in seconds. PP4's derived throughput was **1.315× TP** in this one matched comparison. Both runner and producer exited 0; all completion checks passed.

This is steady host chunk cadence, not per-request latency or TTFT. KV-only final-layer execution omits full final attention/FFN and the output head. One run per topology does not establish repeatability, long-context performance, or a general speedup.

## TP operation profile

Eight profiling requests completed. Coverage validated 71 trace segments, 8 replay sessions, and 32 devices. The final replay contained 69,888 rows. Unmerged `tt-perf-report` retained every duration and its per-device totals matched direct source CSV sums.

| Operation | Mean accumulated kernel duration per device |
| --- | ---: |
| Combine | 18.540 ms |
| ReduceScatter | 18.155 ms |
| Routed expert FFN | 18.039 ms |
| Dispatch | 16.799 ms |
| Matmul | 13.274 ms |
| RingJointSDPA | 9.498 ms |
| ReduceScatterMinimalAsync | 8.981 ms |

Total accumulated kernel work averaged 129.161 ms/device, range 125.210–131.162 ms. These sums can include overlap and are **not elapsed latency**. The final replay subset excludes untraced metadata, socket work, and waits.

The leading investigation targets are dispatch/combine (35.339 ms combined mean-device sum), expert FFN, and reduce-scatter variants (27.137 ms combined). Collective duration may include input-readiness skew; these numbers do not isolate network transport or demonstrate that changing a kernel will improve throughput. Custom MoE operations were incompletely categorized by the analyzer.

## PP4 operation profile: partial

All four ranks executed eight requests and all capture wrappers exited 0. Runtime trace counts were 28/28/28/26 for ranks 0/1/2/3. Ranks 0–2 passed complete eight-replay, eight-device coverage checks. Their mean accumulated kernel sums/device were 75.527/83.037/80.404 ms; these are not stage elapsed times and cannot establish complete pipeline balance.

Two reporting limitations were found:

1. `tt-perf-report` 1.3.0 blanked some valid matmul durations when the inherited operation-to-operation gap was sufficiently negative. Original durations were retained and used for direct totals. Overlapping subdevice work makes the analyzer's rejection unsuitable as evidence of lost kernel timing.
2. Rank 3's host metadata contained 26 trace IDs replayed eight times, but its last trace was empty on all eight devices. Its C++ report attributed only the first session for the nonempty traces. The leading explanation is an empty-trace mismatch between host replay numbering and device trace counters. The failure precedes the custom offline join. No final-replay rank-3 result was accepted or guessed.

The next profiler-specific step is a tiny nonempty/empty trace reproduction with two replays, followed by a narrowly validated fix if confirmed. Full PP4 per-operation claims require repaired attribution and fresh or demonstrably recoverable primary data.

## Hardware and execution status

Following the user's reset, discovery showed 32 devices, 128 links, and 64 neighbor pairs with two channels per pair, matching an 8×4 torus. The Tracy smoke, full-mesh initialization, and PP4 data-transfer probe passed. Both baselines and captures then ran without recurrence of the connectivity failure. Devices were idle at the end of that session; this is not a new September 11 hardware-health check.

## B1 handoff

Task B1 is Sonnet's per-layer MoE variance investigation, with a written finding and issue rather than PR code. Its surviving historical captures can be analyzed without silicon. See [B1 finding](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/B1_FINDING.md) and [local issue draft](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/B1_ISSUE_DRAFT.md). B1's older eager measurements must not be mixed numerically with the September 10 traced baselines above.

No issue has been posted and no message has been sent to Alina.
