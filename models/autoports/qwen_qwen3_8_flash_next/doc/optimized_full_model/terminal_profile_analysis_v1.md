# Selected terminal profile analysis v1

The selected full model takes **113.881454 ms/token**. Independent clean
component traces estimate **115.048771 ms** for the 35/1/12 weighted stack
plus entry/terminal/sampler. The full model is 1.0146%
below that representative estimate: there is no observed >10–15% orchestration
gap. The remaining strict functional-floor gate still fails: useful weights
plus short-window KV give 6.414528 ms, requiring <=64.145280 ms for 10% of that
roofline. **49.736174 ms/token still needs removal.** Intermediate writes and
capacity-wide padding traffic cannot increase the useful-work denominator.

This is a CPU-only analysis of completed, selected-source profiling artifacts.
It does not accept pending projection, norm, residual-write or rank-sum changes.

## Scope, coverage and reproducibility

Run recipe: `run_terminal_final_v1.sh`; driver:
`../../tests/profile_full_model_terminal.py --profile-ops-only --profile-prompt-length 128
--warmup 4 --iterations 1 --capacity 262144` (the recipe is the exact executable
command). The reduced model uses layers 0/1/3 on the serialized 1x4 Blackhole mesh,
B1 with 32 physical sampler rows. A real natural 128-token prefix is staged
outside signposts; the measured decode advances all ranks from 132 to 133. This
includes 33 complete QSA groups and two live compressed-key tiles. Model and
sampler bodies run eager with no live traces. This is one warmed sample per
region, not a statistical throughput run or all 48-layer profile.

`terminal_final_perf_v1/coverage.json` proves identical host/device/final keys
for 72,520 operations across preparation and measurement, all 24 signposts, equal
per-rank region counts, positive matching kernel durations, and no trace IDs.
The eight model leaves partition 1,247 operations per rank. `PERF_TERMINAL`
is exactly final mixer plus head; never add either enclosing model/terminal
region to its leaves. The two sampler strategies use the same frozen real
logits and both select global-max token 1292 (logit 4.34375).

Source hashes are in `terminal_final_ops_v1.json`, including native kernels.
The companion `terminal_profile_analysis_v1.json` records hashes of all input
CSVs/reports, raw geometry, row ranges, budgets and machine-readable arithmetic.
Postprocessing completed successfully; this analysis uses final CSVs and final
tt-perf-report advice, not provisional raw-log estimates.

## Per-region final tables and accounting

| Region | Ops/rank | Merged device sum, ms | Raw rank sum min–max, ms | Full-path weight |
|---|---:|---:|---:|---:|
| token_entry | 2 | 0.002360 | 0.002237–0.002351 | 1 |
| embedding | 3 | 0.037023 | 0.034773–0.039921 | 1 |
| layer0 | 293 | 2.039263 | 1.911158–1.988885 | 35 |
| layer1 | 497 | 3.025558 | 2.899782–2.962819 | 1 |
| layer3 | 403 | 3.041314 | 2.899375–3.000773 | 12 |
| final_mixer | 44 | 0.369678 | 0.358615–0.364080 | 1 |
| lm_head | 3 | 0.656844 | 0.654592–0.656237 | 1 |
| row_assembly | 2 | 0.045847 | 0.044398–0.045367 | 1 |
| split_greedy | 31 | 0.428327 | 0.425441–0.427470 | 1 |
| force_argmax | 15 | 2.885631 | 2.750796–2.807268 | alternative/enclosing |
| model | 1247 | 9.217887 | 8.878836–9.018118 | alternative/enclosing |

The report merges ordinary operations with their slowest rank and collective
operations with a rank mean. Its selected global-call ID can belong to any
rank; geometry joins use that ID, not device0 only. Independently accumulated
raw rank totals retain actual rank identities. Weighted model leaves plus
split sampler give **112.435613 ms** under report merging, versus
per-rank totals **108.827839, 106.650797, 107.942523, 109.471969
ms** for ranks 0..3. These are device-operation sums, not trace-wall measurements
and not a functional lower bound. Eager host/op gaps include profiler and
dispatch overhead; adding them to the clean trace would double-count overhead.

`component_trace_v1.json` separately measures clean, nonblocking 127-replay
windows, with frozen captured inputs, persistent states, initial-position
restores outside windows, exact first-replay checks, input-integrity checks and
stable program-cache entries. Its weighted decoder stack is
**113.436162 ms**, entry/terminal/sampler
**1.612609 ms**. Sum of
reduced components is 9.460295
ms versus combined reduced-model trace 9.397529
ms. Reduced residency and frozen input differ from full 48 autoregression;
therefore this is a representative stack estimate, not an exact measured all48
lower bound. It resolves the orchestration-gap question within those limits.

## Sampler and terminal findings

Split greedy costs 0.428328 ms of merged device kernels and the separately
measured clean component trace costs 0.481792 ms. The kernel sum is only
0.376% of full token latency.
`TopkLargeIndicesDeviceOperation` is 0.266537 ms on 110 cores, processing
row-major BF16 `[1,1,32,62080]` with physical k=32. That kernel dominates the
small sampler region, **not token-out decode**. There is no generic
`TopKDeviceOperation` in this region. Two synchronous two-link ring gathers
collect candidate values/UINT16 indices from `[1,1,32,32]` to
`[1,1,32,128]`, about 6.64/6.96 us each. They are candidate gathers, not a full
vocabulary all-gather. Sampling remains device-side and top-k/top-p capable.

Force-argmax is 2.885631 ms of device kernels: full-vocabulary async gather
1.440606 ms and argmax 1.145874 ms. It is an intentionally measured alternative,
not selected. Both paths are correct for the real frozen logits; profiler host
times 1.469/3.253 ms are not clean trace benchmarks. The selected traced sampler
and token feedback are accounted separately from teacher-forcing model timing.

Final mixer costs 0.369678 ms, head 0.656844 ms, and row assembly 0.045847 ms.
The head's 643.044 us matmul is near its modeled BF16 weight-bandwidth limit
(report 494 GB/s,96.5% of 512 GB/s); it cannot account for a 50 ms deficit.

## Precision, layout and program geometry readback

The raw CSV is authoritative for configuration. Key global-call IDs are
retained verbatim in the companion JSON.

| Operation / call ID | Actual input/output and kernel/config |
|---|---|
| GR down 17071104/17072128 | FP32 `[1,1,1,10240]` times FP32 `[1,1,96,10240]`; materialized FP32 products on 110 cores, accurate SFPU W-sum on 3 cores. HiFi4, approximation off, FP32 destination, `use_sfpu_reduce=true`. |
| GDN out 17213440/17214464 | FP32 activation 6144, BF16 weights 640x6144, FP32 products on 110 cores and accurate W-sum on 20. No lower precision substituted. |
| Ordered QSA indexer 17987586 | BF16 input/weights, FP32 partial output, native `parallel_ordered_projection_compute.cpp`, 100 cores,438.671 us. |
| Live causal scorer 18103296 | BF16 `[1,4,1,128]` query and `[1,1,65536,128]` key storage; native `causal_score_packed_q1_compute.cpp`, 110 cores; packed FP32 `[1,1,4,65536]` logical output,73.673 us. Device position bounds score computation; physical output remains capacity-sized. |
| Sparse KV gather 18110466 | BF16 paged K/V `[8192,1,32,256]` to `[1,1,2080,256]`, 65 reader/writer cores,238.791 us, no compute kernel. |
| QSA SDPA 18118656 | BF16 Q `[1,6,1,256]`, selected K/V 2080 slots; HiFi2, FP32 destination, approximation off, 8x8 grid, qchunk=32/kchunk128, max 16 cores/head,138.512 us. |
| LM head 18326529 | BF16 2560x62080 rank-local vocabulary weight in DRAM width shards; activation/output L1 width shards on 20 cores, output shard 32x3104. HiFi4, FP32 destination, approximation off; block 2, per_core_M 1/N97, 3 workers/DRAM bank, multicore pipeline. |

LM-head raw `CORE COUNT` is 80 (the profiler reports the program resource union),
whereas tt-perf-report displays 8 using its DRAM-sharded heuristic. Its 143%
FLOP-utilization estimate is therefore not a physical achieved-utilization
claim. Use raw configuration and source for geometry; do not infer an 8-core
head or change fidelity based on that figure. BF16 activations/KV/CCL, exact
FP32 GR/GDN arithmetic, selected HiFi4 ordinary projections, inherited HiFi2
SDPA and routed BFP4 policy are preserved. Four full 2560-feature residual
streams remain rank-owned; a 640-feature norm would change the model.

## Largest actionable groups

Budgets below are weighted merged kernel sums. They are **impossible zero-cost
ceilings**, not predicted speedups. JSON records zero-based half-open data-row
ranges in each region CSV, excluding the header. Nested product/reduce budgets
must not be added again to their enclosing local chains.

| Candidate group | Current full-token budget, ms | Disposition / exactness constraint |
|---|---:|---|
| GR down local chain | 11.434657 | Fuse multiply + accurate reduction; product/reduce subset 10.383131. Retain FP32 weights, sequential K-tile fold and SFPU row tree; do not use FMA/K-split reassociation. |
| GR up local chain | 5.174094 | Same exact arithmetic; product/reduce subset 4.424137. |
| GDN output local chain | 6.577142 | Product/reduce subset 6.124910; BF16 weights but FP32 products and accumulation. |
| GR compact slices/concat | 3.334706 | Write existing compact layout directly without changing stream ownership or BF16 rounding boundaries. |
| GR post-gather local reduction/scale/casts | 4.412473 | Keep selected transport, full exact rank-reduction tree and BF16 cast boundary; local padding-aware fusion. |
| GR local norm | 5.744872 | Preserve full local 2560-feature sum tree and FP32 operations. |
| Residual writes | 2.731186 | Fuse cast/multiply/cast and cast/add/cast with explicit intermediate BF16 RNE; no new stream. |
| QSA ordered indexer projection | 5.264052 | Already native; optimize load reuse/row occupancy with its existing tree, not ordinary matmul. |
| GDN QKV | 4.779234 | Already native exact projection,80 cores; retain 16-way and 5-group tree. |
| GDN recurrence | 2.549591 | 12-core execution; candidate finer value-row tasks must preserve per-state-row chronology. |
| MoE fused SwiGLU | 5.552740 | 80 cores, routed BFP4; investigate occupancy without changing accepted dtype or active-expert contract. |
| QSA scorer + reduction + selector | 3.837540 | Native live-prefix dot still feeds capacity-wide ReLU/clone/fill/head-sum/scale; fuse exact selected head reduction and zero inactive storage. |
| QSA sparse KV gather | 2.865492 | 65 data-movement cores, contiguous live-token reuse and sentinel-fill overhead warrant isolated reader profiling. |
| QSA SDPA | 1.662144 |Fixed 2080 slots; any dynamic-position optimization must retain exact mask/selected IDs and trace geometry. |

The first five disjoint groups total
30.933072
ms even if eliminated entirely. Adding norm and writes yields
39.409130
ms, still below the 49.736174 ms requirement and with real mandatory work
remaining. These are high-priority experiments, not a credible promise that
one fusion closes the entire gate.

The selected native causal scorer removes the old capacity-wide dot-product
math. Its **remaining** Q1 score pipeline is rows 230:237 of the layer3 CSV:
73.673 us scorer,44.207 ReLU,46.307 clone,46.307 fill,
44.013 head sum,47.202 scale,18.086 selector. These costs are already in this
profile and are not obsolete scorer traffic. Fusing the exact head reduction
avoids several capacity-sized intermediates. A runtime short-prefix fast path
can select every complete group when <=512 groups exist, but must be
device-driven, preserve selector order/sentinel/tail semantics, and cross the
512-group boundary inside an existing trace without host bucketing/recapture.
The existing all-position correctness probes do not automatically validate a
new fused head-reduction or selector implementation.

## Advice disposition and remaining evidence

* **High op-to-op gaps:** final tables recommend tracing. The measured serving
  path already uses model+sampler traces, persistent inputs and nonblocking
  replay. Component/whole-model reconciliation above confirms this is not an
  unclosed 50 ms host gap. Do not claim eager gap sum as achievable extra speedup.
* **LM head:** report calls the selected configuration optimized. Keep 20/3/block 2
  and its independent geometry sweep; the distorted FLOP% does not justify a
  policy change.
* **Lower fidelity suggestions:** explicit user constraint and inherited
  rejection ledger override generic HiFi2 advice. Ordinary HiFi4 projections
  cannot be downgraded here. No broad precision frontier belongs to this stage.
* **Input0 in L1 / larger block / output subblock:** still actionable for
  flagged small matmuls, including PLE projections and narrow MoE/shared
  projections. They are untried in this report. Block geometry must fit K;
  K=32 cannot blindly use block_w=2. Preserve fidelity and validate exact or
  accepted full-model numerical behavior for each admissible configuration.
* **Unclassified GenericOp/reshape/scatter/SwiGLU:** table categorization is not
  a kernel audit. Raw kernel readback and the group analysis above supply the
  missing attribution. Fresh exact native projection/norm/write/rank-sum
  experiments are pending; none is marked accepted by this report.
* **CCL:** weighted selected kernel transport is about 6.26 ms including entry
  and sampler (exact categories in JSON); local reduction/clone/fill costs are
  separate. Keep selected synchronous two-link ring and inherited rejection
  ledger. Network alone cannot explain the remaining 49.7 ms.

Final advice tables are present and reviewed. Applicable geometry/placement
experiments listed as untried remain work or Stop-Rule hand-off notes; this
analysis does not assert the optimization checklist or strict functional gate
has passed. After any kept candidate, rerun correctness and clean full-model
timing, then refresh component/profile evidence as required by changed scope.
