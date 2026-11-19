# Final B1 profiler findings

CPU analysis of the selected-source S32/B1 captures in `tracy/layer_0` and
`tracy/layer_3`, compared with the preserved global-dispatch/GU-block16 captures
in `profiling_global_dispatch/tracy`. Both final correctness artifacts pass:
linear prefill/minimum decode PCC 0.999988819/0.999997260; full
0.999972759/0.999989250. The capture retains selected DRAM GU input, GU block64,
local-only grouped dispatch, and the original dtype/fidelity policy.

## Four-rank accounting

Raw rows have positive kernel durations on all four ranks. All 16 decode
replays have identical operation signatures within each rank. Linear/full
prefill now have 1546/73 operations per rank, versus 1545/72 historically;
the added operation masks counts before the local expert prefix sum. Decode
remains 177/79 operations per rank per replay.

| Layer / phase | Rank 0 ms | Rank 1 ms | Rank 2 ms | Rank 3 ms | Critical-rank sum ms |
|---|---:|---:|---:|---:|---:|
| Linear prefill | 33.536263 | 33.642886 | 33.528054 | 33.495089 | 33.642886 |
| Linear decode | 0.869292 | 0.867168 | 0.866765 | 0.865746 | 0.869292 |
| Full prefill | 1.205209 | 1.030754 | 1.186624 | 1.216541 | 1.216541 |
| Full decode | 0.352727 | 0.351510 | 0.350910 | 0.348044 | 0.352727 |

Decode values average 16 per-rank replay sums. The maximum rank mean is a
kernel-work proxy, not measured mesh latency or a latency bound: kernel durations omit
dispatch gaps and can overlap. Never sum the four chips as latency. Clean
warmed speedup and efficiency belong to the separate final timing artifacts.

The human and CSV `tt-perf-report` tables merge ranks: noncollectives select
the maximum-duration rank; AllGather/ReduceScatter/AllReduce average ranks and
use the first rank's label. Their `Device` column is not a rank workload.
Final linear decode has 2832 merged rows with representative counts
770/671/801/590; raw data has 2832 rows on **each** rank. Full decode has
1264 merged rows, versus 1264 on each raw rank. Report `Total %` and op gaps
include profiler dumps between replays and cannot be interpreted as clean
latency shares. Human tables were preserved unchanged.

## Changes visible in actual kernels

The following comparisons use the same rank for each old/new pair: rank 0
for decode, rank 1 for linear prefill, and rank 3 for full prefill. Units are
microseconds per measured pass/replay. These are historical-versus-final
profile comparisons, not an isolated causal timing experiment.

| Layer / kernel | Historical | Final | Duration reduction |
|---|---:|---:|---:|
| Linear indexed GU | 49.370 | 36.194 | +26.69% |
| Linear indexed down | 11.579 | 11.629 | -0.43% |
| Linear prefill dispatch | 39.099 | 20.470 | +47.65% |
| Linear prefill combine | 758.562 | 228.710 | +69.85% |
| Linear prefill grouped experts | 541.451 | 544.447 | -0.55% |
| Full indexed GU | 49.311 | 36.108 | +26.77% |
| Full indexed down | 11.630 | 11.575 | +0.46% |
| Full prefill dispatch | 39.455 | 20.079 | +49.11% |
| Full prefill combine | 744.827 | 210.133 | +71.79% |
| Full prefill grouped experts | 495.439 | 495.310 | +0.03% |

The raw sparse rows and report configuration show GU K2048/N256, eight
cores, K-block64, subblock1x1, BF16 DRAM input and BFP8 weights with LoFi.
Down remains K128/N2048, 64 cores, block4/subblock1x1, BF16 L1 input and BFP8
weights with LoFi. The report explicitly identifies active=8/256; the selected
execution still computes gate-selected experts. GU falls approximately 13.2 us
in both kinds, while down is essentially unchanged. The precision-locked
[sparse screen](sparse_geometry/search.json) supplies the isolated selection
comparison; the [geometry audit](local_geometry_audit.md) retains all rejected
measured L1-input, smaller-grid and subblock candidates.

Local-only dispatch cuts combine by approximately 530 us and dispatch by
19 us in both kinds. The per-rank capacity at S32 is 2240 instead of 8192;
global expert IDs are preserved while nonlocal counts become zero. Combine
still initializes all `[1,1,32,8,2048]` output slots with `init_zeros=True`.
Grouped expert compute is effectively unchanged, consistent with compacting
movement rather than deleting selected expert work. Whole critical-rank
prefill sums fall 34.182578 -> 33.642886 ms (linear) and
1.764924 -> 1.216541 ms (full). B32 has separate measured tradeoffs and must not
inherit these B1 percentages.

Raw ordinary projections retain BFP8/HiFi2 for full QKVG/output and shared
projections; router weights are FP32 with HiFi4. Linear prefill output remains
BF16/HiFi4. Grouped expert rows report BFP8/LoFi. Legacy baseline keys in JSON
policy reports are not evidence for local GU input memory or block/core counts;
the raw program attributes, tensor metadata and selected source establish them.

## Remaining costs and report advice

Critical-rank category totals in microseconds per pass/replay:

| Layer / phase | CCL | DRAM-weight matmul | Other compute | Movement/conversion |
|---|---:|---:|---:|---:|
| Linear prefill | 63.076 | 78.813 | 30422.287 | 3078.710 |
| Linear decode | 25.827 | 59.004 | 627.608 | 156.853 |
| Full prefill | 75.684 | 118.266 | 704.177 | 318.414 |
| Full decode | 20.926 | 107.511 | 154.279 | 70.011 |

“DRAM-weight matmul” describes weight placement, not proven bandwidth
saturation. Dispatch/Combine and Typecast are movement/conversion;
PostCombineReduce and SDPA are compute. Explicit SDPA classification corrects
an earlier substring collision with `pad`; durations are unchanged. GenericOp remains compute because the CSV
cannot distinguish all inline arithmetic and internal movement.

- Linear prefill still spends 28000.309 us in GenericOp, over 83% of its
  critical-rank kernel sum. Its ordinary DRAM-weight matmuls total only
  78.813 us. Replacing sensitive ordered helpers needs separate arithmetic and
  correctness evidence; ordinary matmul advice does not qualify such a change.
- Full prefill now spends 495.310 us in grouped experts and 210.133 us in
  combine. Combine no longer dominates the phase, though it remains a visible
  movement cost. Assess grouped work with the natural B32 profile before
  selecting another implementation family.
- TopK remains 43.512/43.473 us in linear/full decode. The full router adds
  25.092 us. Router precision and selection cannot be changed based solely on
  generic fidelity advice.
- Linear decode CCL is 25.827 us: FP32 attention all-reduce 17.546 us and BF16
  MoE all-reduce 8.281 us. Full decode's two BF16 all-reduces total 20.926 us.
  Full decode movement is 70.011 us, so these profiles do not establish CCL
  as the principal remaining cost. Previously measured fused/alternate
  collectives remain documented in the mesh plan.
- Full decode QKVG/output costs 15.920/7.634 us; shared packed/down costs
  8.522/2.659 us. The dense screen measured readers, K blocks and storage-core
  variants. Its near-tied alternatives and unmeasured combinations remain
  explicitly distinguished; there is no exhaustive optimality claim.
- The report continues recommending L1 GU input and wider sparse subblocks.
  Those specific legal layout families were measured and slower in the local
  screen. Do not reclassify generic advice as a missing experiment or change
  fidelity automatically. Unmeasured combinations remain unmeasured.

Exact final artifacts: `tracy/layer_{0,3}/{ops.csv,ops_source.txt,correctness.json,correctness.json.provenance.json}`
and `tracy/layer_{0,3}/{prefill,decode}_perf_report.{csv,txt}`. Reconstructed
per-rank categories and replay signatures use `summarize_multichip_perf.py`;
`perf_summary.json` and phase `*_op_groups.csv` are emitted by the separate
summary command. Historical counterparts are under `profiling_global_dispatch/`.
This analysis opened no devices and changed no implementation, checks or raw
profiling artifacts.
