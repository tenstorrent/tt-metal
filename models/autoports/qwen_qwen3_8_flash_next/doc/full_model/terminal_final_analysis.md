# Current terminal and QSA operation attribution

The final C262144 reduced profile passes complete coverage and both common
samplers return the actual greedy token 243975 on the same frozen real logits.
**Split greedy remains selected:** its merged kernel sum is 0.429263 ms versus
2.886563 ms for force-argmax (6.724× slower). Split accounts for 3.673% of the
reduced model-plus-split kernel sum. The evidence does not show a dominant
sampler requiring another terminal rewrite.

This is layers0/1/3, B1, one warmed **eager** operation-profile iteration, with
two warmups per region. It preserves the full262144 allocation and scorer width,
but starts with a very short live history. It is not an all48 model trace,
maximum-position performance test, or advertised throughput. The all48 clean
138.754269 ms token-out measurement remains authoritative for that workload.

## Coverage, units and boundaries

`terminal_final_ops_v1.json` passes with current source hashes and no live traces.
All 39 captured source hashes match the files inspected for this analysis.
`terminal_final_perf_v1/coverage.json` was independently recomputed from the raw
host messages, device CSV and final report, reproducing the saved result exactly:
21488 device operations match on host/device/final output; no rows are skipped;
all six signposts survive unchanged. Every measured duration is positive and
preserved, and each region has equal operation counts on all four ranks.
The final CSV SHA256 is
`055fbc6cbcada681661804279df2f26dd3187978b0eee42280543b9cf87f8854`.
Raw captures and historical failed attempts remain untouched.

| Region | Ops/rank | Per-rank kernel sums, ms | Merged kernel sum, ms | Signpost wall window, ms |
|---|---:|---|---:|---:|
| Reduced model | 1253 | 10.844644 / 10.889831 / 11.034911 / 10.954305 | 11.256808 | 34.940769 |
| Split greedy wrapper | 31 | 0.428101 / 0.428307 / 0.425588 / 0.424657 | 0.429263 | 2.241439 |
| Force-argmax wrapper | 15 | 2.957070 / 2.959557 / 2.794982 / 2.794464 | 2.886563 | 4.209090 |

Raw device durations are **nanoseconds**; `tt-perf-report`'s `Device Time [μs]`
is microseconds. Tables here divide those values by 1000. Version1.5.0 merges
noncollectives using the maximum rank duration and AllGather/ReduceScatter/
AllReduce using the mean rank duration (`merge_device_rows`). Thus the sum of
merged rows is neither a single-rank wall time nor the sum of all four devices.
The report's `Total [%]` also includes operation-gap accounting; the sampler
fraction above is explicitly calculated from kernel sums, not that column.
Eager host gaps, signpost overhead and profiler instrumentation must not be
substituted for clean traced latency.

## Named costs in the current production path

These are source-order attributions within the measured model region. Exact
zero-based half-open row intervals, first/last global call IDs, artifact hashes
and largest operations are in `terminal_final_analysis.json`.

| Boundary | Merged kernel sum, ms | Source / scope |
|---|---:|---|
| Device token slicing and PLE token-word formation | 0.002304 | `_decode_body`, before embedding |
| Embedding, tilization and width gather | 0.036766 | `QwenModel.embed` |
| One QSA layer's full-capacity indexer score graph | 2.387150 | `fused_indexer_scores`, query/key casts through final scale |
| QSA causal selector | 0.010177 | `fused_selected_token_ids`, native heap over complete groups |
| QSA selected K/V gather | 0.062469 | `local_selected_paged_kv`, fixed2080 slots |
| Final norm/hyperconnection mixer | 0.372330 | `FinalMixer.__call__`, including its CCL and exact projections |
| Vocabulary head including layout conversions | 0.668237 | 0.000933 input shard +0.653647 DRAM matmul +0.013657 output interleave |
| Sampler row fill and valid-row write | 0.045470 | 0.024106 native fill +0.021364 row writer |

The scorer's largest operation is its FP32 product at0.908385 ms, followed by
FP32 width reduction0.367437 ms. Key BF16→FP32 cast is0.140156 ms;
reshape0.212630 ms, ReLU0.171170 ms, clone0.170419 ms and padding fill0.172099 ms
are also material. These are existing decoder costs exposed by the allocated
65536 compressed-key width, not full-model wrapper or sampling operations.
The ordered indexer projection itself is0.438237 ms
(`parallel_ordered_projection_compute.cpp`), essentially unchanged from the
historical0.438841 ms. QSA SDPA is0.141283 ms and retains its selected BF16/HiFi2
policy; it is not a full262144-token dense-attention operation.

The LM-head row confirms BF16×BF16→BF16, HiFi4, FP32 DEST enabled,
`math_approx_mode=0`, `packer_l1_acc=0`, direct DRAM-sharded matmul block1 with
three workers per DRAM bank. The raw compute source is
`bmm_dram_sharded_multicore.cpp`; output staging is width-sharded L1 followed by
DRAM interleave. No lower-fidelity projection is inferred from the faster time.
The scorer width reduction records FP32 input/output, SFPU reduction, HiFi2,
FP32 DEST enabled, approximation and packer accumulation disabled. This is the
existing selected QSA configuration. Final mixing retains its exact FP32
projection and reduction boundaries with BF16 activation boundaries.

One metadata limitation must remain visible: some `BinaryNgDeviceOperation`
shape annotations are inconsistent with their invocation. For example the
908.385 μs scorer product (global call 5129219) lists a 10240-wide projection
shape, while its preceding key cast and following reduction/reshape explicitly
record 65536 keys; the frozen source dispatch order identifies that product.
The historical C128 profile has the same annotation pattern. This is consistent
with reused profiler metadata, but its parser/cache origin was not diagnosed in
this audit. Operator duration/coverage and adjacent shape evidence are intact.
Do not calculate product FLOPs or bandwidth from those anomalous BinaryNg
shape fields. The component attribution uses source order and the surrounding
valid tensor metadata, not the inconsistent annotations.

## Sampler semantics and advice

Both samplers share identical frozen real logits and return243975, equal to the
reference argmax calculated **outside** measured regions. Greedy parameters are
top_k1, top_p0, temperature1; physical top-k staging width32 does not make this
stochastic sampling. This is one actual-logit greedy check, supplemented by the
existing trace-feedback and sampling-contract evidence, not an exhaustive tie
semantics test.

Split's largest kernels are TopkLargeIndices0.266540 ms, route prep0.024541 ms,
route finish0.020419 ms, sampling0.020802 ms and manual seed0.018544 ms.
Force-argmax instead pays1.442554 ms for vocabulary AllGatherAsync and1.144347 ms
for ArgMax, plus0.112481 ms concat. The current device seed/token-feedback
helpers are inside each wrapper. Neither measured wrapper performs host
argmax or full-logits readback. The caller token read is outside the profile's
sampler signpost; its cost is represented separately by the clean token-out
acceptance boundary.

Advice-enabled reports contain **no head or sampler advice**. Twenty remaining
model advice rows concern inherited decoder matmuls: activation placement in L1,
block/subblock sizing, and generic HiFi2 suggestions. Decoder projection
geometry and precision were already selected/rejected at stage5; a generic
lower-fidelity suggestion does not authorize overriding those policies. This
analysis selects no decoder tuning, sampler replacement or capability reduction.
The earlier head advice to consider DRAM sharding has been implemented and is
absent from the new head row.

## Historical comparison and limits

The same 12-operation scorer section in historical C128 `terminal_perf_v3`
totaled0.025220 ms, versus2.387150 ms now: a2.361930 ms increase for the one
profiled QSA layer. The actual key-cast/reduction metadata changes from 32 to
65536 compressed keys. The native selector remains0.010167→0.010177 ms and
selected-K/V gather0.062070→0.062469 ms on these short histories. This strongly
supports the capacity-dependent scorer explanation and rejects attributing the
full-model gap to a full-capacity native heap scan.

The old head matmul was0.809006 ms; the current DRAM head is0.653647 ms. Old
row assembly included32 untilizes plus concat/tilize/copy (about0.608333 ms);
current native fill/write is0.045470 ms. Old split wrapper was65 ops and
0.467541 ms; current is31 ops and0.429263 ms. These are historical comparisons
across terminal revisions, not paired current-code ablations. The reduced model
sum rises9.440300→11.256808 ms despite terminal savings because the large-context
scoring work is now represented.

A reduced eager op sum cannot quantify the exact28.453660 ms all48 token-out
versus historical weighted-decoder gap. Do not multiply the scorer delta by 12
and label the result measured all48 latency: execution mode, resident memory,
actual token histories and selected rank timing differ. The measured named
costs, unchanged decoder policy and current clean full-model counters make a
wrapper/sampler explanation unsupported; an exact full48 decomposition remains
unmeasured. No additional full48 profiler run or decoder sweep is proposed by
this audit.
