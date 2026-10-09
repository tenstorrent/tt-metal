# BFP8 bandwidth and bottleneck audit, October 9

Recomputed from all four completed B16/B32 native/shared-QK Tracy captures,
checking raw CSV hashes and signpost-window row/time accounting on all TP ranks.
These are warm eager two-layer profiles with real weights and synthetic caches.
They are not complete traced-model attribution or NoC hardware counters.

| 32K context, users/TP4 | Native measured TSU | Useful bytes / step / chip | Modeled effective GB/s | Fraction of ideal | Ideal bandwidth-only TSU |
|---|---:|---:|---:|---:|---:|
| 16 | 11.75 | 12.884 GB | 151.37 | 29.57% | 39.74 |
| 32 | 7.35 | 18.655 GB | 137.19 | 26.79% | 27.45 |

Assumes 512 GB/s/chip, one padded BFP8 weight read per step, BFP8 KV and one
FP32 recurrent-state read/write. Excludes compute, CCL, extra traffic and launch
costs. This ratio is modeled useful traffic / observed decode time, not measured
DRAM utilization. The projected shared-QK plus epilogue paths would reach
37.35%/38.31%, but their full-model throughput has not yet been measured.
At 32K, 30 TSU is below the ideal B16 ceiling but exceeds this B32 traffic
model's ceiling. A changed memory/TP design is needed for that B32 target.

Tracy at B32: optimized recurrence/preparation sums 375.63 us versus 1160.98 us
native; the largest generic recurrence kernel is about 159 us on rank 0.
Preparation and surrounding ops remain substantial. Packed convolution sums
149.71 us, RoPE 109.46 us and paged attention 1504.90 us (median across ranks).
Inclusive stages overlap and must not be added as disjoint costs. The 48 GDN
and 16 attention layers also have different multiplicities.

All selected rows have blank NoC utilization, DRAM utilization, congestion,
compute CB-wait and per-core min/max columns. RISC durations include waiting;
they cannot distinguish memory supply, NoC contention, backpressure or compute.
The new [phase profiler](../gdn-phase-profile-launch-v2/README.md) records
explicit reader/compute/writer regions with one/two input buffers. It does not
change the serving source and instrumentation times are diagnostic only.

Earlier ~20 TSU was the 19.15 mean from the optimized BFP4 GPQA run. Native
recurrence then measured 15.03, and the accepted BFP8/corrected-harness run
13.01. Those variable-length scores are not a controlled perf comparison.
The isolated fixed-context B16 precision comparison finds BFP4/HiFi2 faster
than BFP8/HiFi2 by 9.3% at16K and 8.6% at32K. Precision alone does not explain
the larger kernel-policy performance change.

`audit.json` retains the input hashes, missing-column counts, stage medians,
all-rank recurrence operation times and the explicit traffic calculation.
Run `reproduce.py --model <model-dir> --output <new-directory>` to reanalyze.

The separately saved native-before snapshot contains a newly completed
32K/B32 cell at7.34297TSU and B16 at11.75371TSU, with two still-queued
16K cells. The overall run was
still active; this partial receipt is not a completed sweep.
