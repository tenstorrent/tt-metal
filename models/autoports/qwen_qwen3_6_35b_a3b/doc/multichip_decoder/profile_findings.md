# Selected B32 profiler findings

CPU analysis of the selected-source, natural-activation captures in
`perf_b32/tracy/layer_{0,3}`. The [B1 findings](profile_findings_b1.md) retain the
historical global-dispatch/GU-block16 comparisons. This document analyzes B32
separately: its grouped EP expert path differs from indexed B1 execution.

## Scope and correctness

The `seeded_b32_v1` fixture profiles **one representative S32 prefill request**
from a batch of 32, followed by **16 whole-batch B32 decode replays**. Prefill
is not a whole-batch measurement. Decode is not divided by 32. State is restored
once before the timing region; its fixed input/position replays then evolve
state for profiling. They are not sixteen independently validated generation
steps. The linked clean seed acceptance covers eight changing decode steps;
the narrow capture checks two steps against both the optimized baseline and
saved mesh outputs, eager/trace agreement, deterministic replay, and all
32 rows × two state tensors × four ranks through actual upload roundtrips.

| Layer | Prefill whole-output PCC | Decode PCC, steps 0 / 1 | Capture passed |
|---|---:|---:|---|
| 0, linear | 0.999973622 | 0.999955139 / 0.999950210 | true |
| 3, full | 0.999977980 | 0.999960752 / 0.999959248 | true |

Exact controls and seed paths are in each `correctness.json`; current-source
provenance is in `correctness.json.provenance.json`. Clean warmed timing,
speedup and efficiency use the separate final acceptance artifacts, not these
instrumented profiler measurements.

## Four-rank kernel accounting

All four ranks have positive device durations. Raw prefill contains 1546
linear or 73 full operations per rank; each decode replay contains 3636 linear
or 1588 full operations per rank. The summary checker validates all sixteen
replay signatures. Values below are sums of kernel durations on each rank;
decode averages sixteen such sums. The maximum rank mean is a kernel-work
proxy, **not mesh latency or a general latency bound**: dispatch gaps are absent
and kernels may overlap. Four ranks must never be added to claim latency.

| Layer / phase | Rank 0 ms | Rank 1 ms | Rank 2 ms | Rank 3 ms | Maximum rank mean ms |
|---|---:|---:|---:|---:|---:|
| Linear, one prefill | 33.641158 | 33.627721 | 33.636469 | 33.637417 | 33.641158 |
| Linear, whole B32 decode | 19.710272 | 19.704679 | 19.730907 | 19.716859 | 19.730907 |
| Full, one prefill | 0.984286 | 0.831271 | 0.961996 | 0.912010 | 0.984286 |
| Full, whole B32 decode | 5.802994 | 5.799921 | 5.796485 | 5.788427 | 5.802994 |

Category sums on the corresponding maximum-work rank, in microseconds:

| Layer / phase | CCL | DRAM-weight matmul | Other compute | Movement/conversion |
|---|---:|---:|---:|---:|
| Linear prefill | 183.231 | 81.808 | 30343.807 | 3032.312 |
| Linear B32 decode | 610.642 | 60.686 | 14577.584 | 4481.995 |
| Full prefill | 47.560 | 120.021 | 556.825 | 259.880 |
| Full B32 decode | 338.154 | 814.190 | 2725.227 | 1925.423 |

DRAM-weight matmul describes placement, not measured bandwidth saturation.
Dispatch, Combine and Typecast are movement/conversion; PostCombineReduce and
SDPA are compute. The explicit SDPA classification corrects an earlier name
substring collision with `pad`; it changes categories, never durations.
GenericOp remains compute in this coarse table although its inline programs
can also contain data movement.

## Actual execution and dominant costs

Linear prefill spends 28004.852 us (83.25% of its rank sum) in GenericOp. Its
ordered FP32 recurrent/chunk helpers dominate; ordinary matmul tuning cannot
account for that work. Linear B32 decode spends 8082.294 us in GenericOp,
3926.984 us in BinaryNg and 1239.297 us in FastReduceNC. The ordered BF16 input
projection helper group costs 3633.828 us across 64 calls; ordered output
projection costs 1712.669 us across 32 calls. Gated norm helpers add 1229.635 us.
Movement includes FillPad 969.450 us, transpose 780.781 us, slice 629.973 us,
permute 413.360 us and Typecast 374.446 us. These are measured reasons to
investigate batching or reducing layout transitions; no faster equivalent
sensitive-arithmetic implementation is established by this profile.

Full B32 decode still executes attention per request before batched MoE.
LayerNorm totals 1099.381 us, including 787.689 us for 32 singleton H2048
normalizations. QKVG and output projections cost 512.673 and 240.259 us over
32 calls each; SDPA costs 452.456 us, rotary 203.369 us, and 64 paged cache
updates 203.196 us. Slice totals 388.834 us. Thus batching attention and
preserving layouts across its existing stages are concrete investigation
families; this report does not claim an implemented or measured improvement.

The final row-output split is outside MoE. In all sixteen replay signatures
it is one Slice followed by 31 `(UntilizeCodegen, Slice, TilizeWithValPadding)`
triplets: 94 operations. On the maximum-work rank it costs 200.190 us for
linear and 200.174 us for full (1.01% and 3.45% of whole-layer kernel work).
The full costs are 32.229 us slicing, 80.166 us untilizing and 87.780 us
tilizing. Keeping a batched tensor through the next layer could avoid this
boundary conversion, but would require a new validated stack/input contract;
these costs remain included in the current whole-layer denominator.

The [source-boundary MoE attribution](moe_share.json) includes router input
conversion, routed and shared experts, and their output collective, while
excluding normalization, final residual and output split. B32 linear MoE is
921.866 us / 4.67% of whole-layer kernel work; full is 838.284 us / 14.45%.
These are kernel-work fractions, not fractions of clean elapsed latency.
They differ markedly from B1's 23.08% / 55.55%; selecting another expert
parallel family solely from B1 would misidentify B32's dominant costs.

CCL totals are 610.642 us linear and 338.154 us full per complete batch.
The 32 attention AllReduce calls contribute 436.522 and 291.591 us respectively.
Linear rows carry FP32 inputs and FP32 accumulation with BF16 output dtype;
full rows carry BF16. Native attributes show Linear topology, axis 1 and
ring size 4. Grouped MoE uses its separate ReduceScatter/AllGather path.
The profile does not establish CCL as the principal remaining cost.

## Precision, geometry and active-expert evidence

| Actual path | Local geometry and policy observed in raw metadata |
|---|---|
| Full singleton QKVG / output | K2048/N2560 and K1024/N2048; BF16 input, BFP8 DRAM-width-sharded weights, HiFi2, FP32 accumulation; DRAM-sharded program K-block4, three workers per bank, per-core N2; L1-sharded output |
| B32 shared packed / down | M32 K2048/N288 and K128/N2048; BF16/BFP8, HiFi2, FP32 accumulation; ordinary 1D multicast, K-block2 and subblock1×1 |
| Router / prefix | BF16 activation with FP32 router weights, HiFi4, FP32 output/accumulation; prefix also preserves FP32 accumulation |
| Grouped experts | 64 local experts, 8×8 grid, `m_tiles=1`, `read_x_at_offset=true`; BFP8 weights, BF16 output, LoFi, **FP32 destination accumulation disabled** in this native grouped kernel |
| Linear attention | Ordered BF16 projection helpers and sensitive FP32 recurrent intermediates, HiFi4; GenericOp rows are not ordinary BFP8 matmul |

Raw dispatch retains global 256-expert IDs and top-8 choices, with group size 1,
axis 0 and zero dispatch network links. Each rank masks nonlocal counts before
prefix construction and skips nonlocal dispatch-table entries. Local capacity
is `round_up(8*32 + 31*64, 32) = 2240` rows, versus 8192 for the historical
256-expert allocation. The grouped kernel consumes device counts and local
expert IDs; allocated 64-expert weights do not mean dense execution of all
experts. See `_grouped_moe` in [the decoder](../../tt/multichip_decoder.py) and
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp`
for count-controlled execution. CSV metadata alone cannot expose the numerical
routing IDs; correctness/routing controls supply that evidence.

Combine has `init_zeros=true` and returns `[1,1,32,8,2048]`, preserving zero
slots for absent/nonlocal routes before weighted reduction. The representative
prefill's grouped kernel/combine costs are 413.068/175.593 us linear and
347.039/148.748 us full. Whole B32 decode costs are 344.335/147.799 us linear
and 374.630/160.330 us full. No indexed SparseMatmul appears in B32 decode:
the selected B1 GU block64 improvement must not be credited to this path.

## Report interpretation and remaining advice

Human reports merge four ranks into representative logical-op rows:
noncollectives choose maximum duration, whereas AllGather/ReduceScatter/
AllReduce average rank durations and retain a representative device label.
The decode tables therefore have 58176 linear and 25408 full rows, while raw
CSV contains that many rows **per rank**. Report Device labels cannot recover
per-rank workload; summing report rows is not the critical-rank calculation.

Report `Total [%]` includes profiler-drain gaps: merged decode gaps total
27.578 s linear and 18.842 s full, versus merged kernel sums 321.326 ms and
95.347 ms over sixteen replays. These percentages are not clean latency shares.
The **overall DRAM roofline has a different denominator**: installed
`tt_perf_report/perf_report.py::calculate_overall_dram_roofline` divides modeled
bytes by `timeline.busy`, excluding those gaps in these reports. Its linear
`0.0% (0 GB/s)` is rounding of only 0.046842% prefill / 0.034836% decode modeled
utilization, not zero traffic and not a profiler-drain denominator. GenericOp
and grouped MoE lack byte/FLOP modeling, so most linear work is absent from the
numerator. Full reports show 2.2% / 7.8%. None is a complete measured DRAM
bandwidth utilization for the decoder.

The full decode report marks 1024 rows FLOP-bound (the 64 attention projections
per replay) and 64 ordinary matmul rows SLOW. Linear has 64 SLOW rows and no
modeled classification for its dominant inline work. Generic advice asks for
L1 input, subblocks wider than 1×1, and fidelity changes. These estimates do not
justify fidelity changes: router/recurrent precision is sensitive.
The [local geometry audit](local_geometry_audit.md),
[dense screen](geometry/search.json) and [sparse screen](sparse_geometry/search.json)
record measured legal alternatives at unchanged policy. They support retaining
the selected singleton geometry and B1 GU64 layout; they do **not** establish
optimal B32 shared/grouped geometry or reject its unmeasured candidates.
The [local-dispatch screen](local_dispatch_l0_true.json) and counterpart false/
full-layer records retain the real whole-B32 prefill tradeoff: linear worsened
about 0.79%, full improved 7.42%; corresponding decode improved 2.16% / 5.89%.
Those paired selection tests are separate from these representative-request
profiles and cannot be substituted for final clean baseline speedup.

Exact artifacts are `perf_b32/tracy/layer_{0,3}/{ops.csv,ops_source.txt,correctness.json,correctness.json.provenance.json}`,
`{prefill,decode}_perf_report.{csv,txt}`, phase `*_op_groups.csv`, and
`perf_b32/perf_summary.json`. CPU postprocessing preserves original device
C++ timings; filtered host export retains TT_DNN/TT_METAL roots and omits
optional child host timings. Its manifest records that limitation. Human
reports remain unchanged. This analysis opened no devices and changed no
implementation or captured artifacts.
