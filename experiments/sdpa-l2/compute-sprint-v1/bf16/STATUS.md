# A/B compute sprint — historical screening status

Superseded by `REPORT.md`, which contains the final retained configurations,
held-out qualification, fresh paired timings and clean profiler results.

Fixed Q256/K512/D128, two K/V slots, original reader/writer and CB formats/counts.
Canonical numerical recipes come directly from `flux2-frontier-v1/device_attention.py`.
All timings below use 16 resident Q repeats × 512 K chunks, 12 warmups,
10 timed samples per implementation, alternating baseline/disabled/candidate order.
Times are blocking trace wall times; not profiler-instrumented. TF/core is useful
QK+PV FLOPs divided by measured time. No clock-dependent utilization claim here.

| Variant/candidate | Canonical ms | Candidate ms | Canonical TF/core | Candidate TF/core | Decision |
|---|---:|---:|---:|---:|---|
| A width cache | 275.7964 | 275.9079 | 1.99334 | 1.99253 | Reject: no gain |
| B width cache | 345.6265 | 345.8458 | 1.59061 | 1.58960 | Reject: no gain |
| B redundant L1 toggles | 345.5675 | 345.6031 | 1.59088 | 1.59071 | Reject: no gain |
| B numerator plane grouping | 345.6417 | 344.3691 | 1.59054 | 1.59641 | Preliminary 0.368% less time |
| A eight-tile exp batching | 275.7946 | 275.2310 | 1.99335 | 1.99743 | Preliminary 0.204% less time |
| A max-stat block copy | 275.8634 | 275.8384 | 1.99286 | 1.99304 | Reject: noise-scale change |

All table outputs are bitwise identical to canonical and candidate-disabled
copies. B candidates also passed distinct K8192 with per-chunk K scale growing
from 0.25 to 4, which changed sampled query maxima 9–13 times after first chunk.
A max-stat block-copy passed this test; performance not measured yet.

Numerator plane grouping is an E/G-agent contribution independently transferred
to B. DST grouping becomes hi0/hi1/lo0/lo1/chunk0/chunk1/correction. The same
15 SFPU arithmetic instructions now use three two-tile copies and two two-tile
packs instead of six scalar copies and four scalar packs. L1 storage unchanged.

Pending candidates: A max-stat block-copy, A repeated denominator-L1-enable
elimination, A eight-tile approximate-exp replay batching, B correction-vector
reuse (even/odd column vectors share the same column-broadcast correction).
No pending candidate should be considered qualified or a measured speedup.

At 2026-09-18 14:27 UTC, A exp batching smoke failed **before device open**:
Ethernet core 31-25 did not become active during firmware initialization.
No candidate compute ran; root inspected/reset reserved hardware under global
lock and fresh matmul smoke passed at 14:31:14 UTC. Raw failure log:
`A-expblock-growing-01.log`. Subsequent jobs use fresh paired controls.

Fullchip correctness adapter is prepared: Q2048/K8192, two cores, four distinct
Q jobs/core, original canonical reader/writer, canonical/private-disabled/
candidate comparison, normal/scaled/outlier/common-mode/growing-max inputs,
two trace replays. B plane-grouping + correction reuse passed all six input
distributions (18 canonical/disabled/candidate records), including both trace
replays. Combined performance is pending.
