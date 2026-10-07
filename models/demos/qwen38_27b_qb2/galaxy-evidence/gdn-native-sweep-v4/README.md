# Completed native control for the matched TP4 sweep

[Graphs and data table](index.html) · [CSV](sweep.csv) · [JSON](sweep.json)

One physical TP4 replica on `10.228.203.98`; all 64 model layers, fixed
128 output tokens, one warmup and three measured runs per cell. Prefill
uses fresh state with no prefix reuse. Decode uses device sampling and
deferred history readback. There is no HTTP/router overhead in these numbers.

The sweep completed 24 measurements and recorded one allocator OOM:
B32 / 32,768 input tokens. Ten further cells have explicit capacity or
implementation guards and were not attempted. `completed_with_oom` means
the sweep finished with that measured failure; it is not a universal pass.
Batch 64 remains unsupported by the full model's projections/token buffers.

| Input tokens | Batch per TP4 | Prefill input tokens/s | Aggregate decode output tokens/s | Decode tokens/s/user |
|---:|---:|---:|---:|---:|
| 128 | 32 | 3,702.46 | 305.51 | 9.55 |
| 8,192 | 32 | 7,168.27 | 289.64 | 9.05 |
| 32,768 | 16 | 6,565.49 | 209.28 | 13.08 |
| 131,072 | 8 | 4,875.90 | 123.58 | 15.45 |
| 262,016 | 4 | 3,677.10 | 65.68 | 16.42 |

These are the **native recurrence control**, with accurate full-tile decode
attention and the common compact decode/prefill knobs. The single-step
candidate is running afterward under the same workload and settings; no
candidate full-model speedup is established by this artifact. These are
one-replica measurements, not extrapolated whole-Galaxy throughput.

The first 12 measurements are reused from the immutable v1 baseline after
validating source/policy hashes, runtime configuration, prompt token hashes,
repeatability and raw timing accounting. `attempt-01` records the B32 / 32K
allocation failure and confirmed cleanup. `attempt-02` runs in a fresh
process, preserves that explicit OOM and finishes the remaining 12 points.
The controller saves each original attempt and exposes the aggregate in
`sweep.json`. Source and persistent launch hashes are in the neighboring
`gdn-sweep-recovery-v1/launch-gdn-perf-sweeps-v4.json`.

Input throughput is input token count divided by directly timed prefill;
decode throughput excludes prefill. The fifth graph separately reports
output throughput over the entire request. Near-256K reserves room for the
128-token output. Graphs include only measured values, and lines connect
measured cells rather than predicting untested points.
