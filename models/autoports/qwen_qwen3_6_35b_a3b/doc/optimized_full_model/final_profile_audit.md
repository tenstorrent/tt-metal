# Final full-path profile audit

CPU-only audit, 2026-11-16. The final reduced native capture and conversion
**pass**. The selected full-model decode has no evidenced >10–15% orchestration
gap and sampling does not dominate it. Native operands/configs agree with the
selected numerical policy and the optimized head/custom kernels. Explicit
sampler all-gather output buffers remain **disabled** after the documented
native correctness failure; decoder persistent CCL scratch remains enabled.
The separate nonaligned context smoke also passes, as scoped below. This audit
is not stage-review acceptance. No TTNN import, hardware operation or
production/test edit was performed.

## Artifact identity and capture validity

The installed `wait_for` observed `DONE profile_driver` in
[final_measurement_driver.log](logs/final_measurement_driver.log), exit0,
after2732s waiting; the driver records2942s for capture/conversion/reporting.
The subsequent [32769-token context smoke](final/prefill_context32769.json)
passes on all40 layers with context capacity65536: public nonaligned prefill
and two queued model/sampling replays, positions advancing to32771, zero
per-step refresh/readback/capture/release operations and stable9590 cache
entries. This is one repeated-AIME-token capacity/transport smoke, not a
32769-token quality evaluation or a65536-token maximum-length run.

The [component summary](final/profile_reduced/component_summary.json) is
`validated_profile`, `headline_eligible=false`: real layers0/3, full embedding,
terminal norm/head and split greedy sampling, B1/context1024, sixteen replays
per component. All four ranks have199 model ops and31 sampling ops per replay;
29,440 native execution rows are validated across the token-out, model-only
and sampler-only windows. All fifteen recorded source hashes match both the
current files and [clean all40 performance](final/performance.json). Native
metadata inventories680 rank/program entries with operands and compute/DM
kernel sources. Empty fidelity lists mean validated DM-only programs.

Exact commands are in [commands.log](final/profile_reduced/commands.log).
The profiler environment enables native device/TTNN/trace profiling and
mid-run drains, with Watcher and allocation tracking unset. The successful
conversion retains:

- [Native host trace](final/profile_reduced/raw/.logs/tracy_profile_log_host.tracy),
  1,684,550,415bytes; [native device executions](final/profile_reduced/raw/.logs/cpp_device_perf_report.csv),
  184,828,059bytes; [native messages](final/profile_reduced/raw/.logs/tracy_ops_data.csv).
- [ops.csv](final/profile_reduced/ops.csv), 2,353,503,058bytes,
  SHA256 `d107024b0d9e0985663bbc63222d29fc296bd1b913f4b3f6647bfaa557d8b3fa`.
- [native_program_metadata.json](final/profile_reduced/native_program_metadata.json),
  SHA256 `54632a4b054086c2a058af8847c758fc02d6d19cb36cc7422bacb51fd12de1d2`.
- [Profile performance](final/profile_reduced/performance.json), SHA256
  `eb6e43d031ef180d73a8b33707d77a847ee7656800d951e7a68404e7c77bda66`.

The optional WASM-copy warning at capture.log:1107 refers to the default
`generated/profiler/.logs` destination. The actual capture above exists and
successfully converted; the warning is not missing native data. No dropped
marker/source-location-overflow warning was found. `tt-perf-report` warns that
several custom/new ops are unclassified; their native timings and metadata are
retained and attributed explicitly here rather than omitted from work totals.

Advice remains enabled in the complete [token-out table](final/profile_reduced/token_out_perf_report.txt)
and [CSV](final/profile_reduced/token_out_perf_report.csv),
[model-only table](final/profile_reduced/model_only_perf_report.txt) and
[CSV](final/profile_reduced/model_only_perf_report.csv), and
[sampler-only table](final/profile_reduced/sampler_only_perf_report.txt) and
[CSV](final/profile_reduced/sampler_only_perf_report.csv). The first two full
tables/CSVs exceed the repository500KiB hook limit and remain untouched raw
workspace artifacts. The three `*_op_groups.csv` files are113129/94604/18769
bytes. [Compact attribution](final_profile_attribution.csv) preserves every
rank2 token-out group and adds bounded source-role labels; no rank is summed
with another to manufacture latency.

## Wall time, operation work and stack budget

| Measurement | ms/step | Scope |
|---|---:|---|
| All40 queued token-out |21.416868244|Clean127-step evolving model+sampling replay;46.692168t/s/u|
| All40 caller-observed token delivery |21.461292819|Includes127 output readbacks;46.595515t/s/u|
| All40 traced HF teacher forcing |20.933991111|99 actual HF token uploads/model replays;47.769200t/s/u; excludes sampling/readback|
| All40 fixed-token model-only |20.934684803|Diagnostic fixed token, advancing position; not teacher forcing|
| All40 sampler-only control |0.487714543|Fixed real logits, token merge/seed advance, no decoder|
| Reduced profiled token-out |2.015464375|Fenced/instrumented real layers0/3|
| Reduced profiled model-only |1.514857939|Separate fixed-token diagnostic|
| Reduced profiled sampler-only |0.573671314|Separate fixed-logit diagnostic|

All40 queued token-out has127 model and127 sampling replays and **zero**
per-token token/position/page/activity uploads, output readbacks, capture or
release operations; the device-only guard passes and program cache remains
8440. Initial/final positions128→255 and final token agree with the separate
observed-delivery run. Warmed TTFT is2729.973400ms, including public prefill,
terminal/sampler, initial position upload and first-token readback; preparation,
compilation and capture are excluded. These boundaries are different from
decoder-only latency and from the reduced profile.

The current [queued-layer engineering budget](final/layer_stack_budget.json) is
`30*0.586715590034 + 10*0.306684234965 = 20.668310050678 ms`.
Whole token-out exceeds it by0.748558193374ms, **3.6218% before allowing any
terminal work**, already below the requested10–15% trigger. These are isolated
real-weight layers with synthetic fixed inputs/router choices and position128,
not a mathematical bound or exact subtraction of full40 decoder work. The
raw runner's24.99350229ms reference is explicitly historical stage5 evidence;
it is not the current optimized estimate.

The [native accounting derivation](native_accounting_derivation.md) and
[ordered-row provenance](native_accounting_derivation.json) independently
reconstruct entry, layer0, layer3, terminal/position handoff and sampling from
the same reduced token-out window. All sixteen per-rank replay signatures
match; all program IDs exist in native metadata and regrouped sums reproduce
the component summary and op-group CSVs. Layer boundaries are inferred from
source and ordered shapes, not measured layer signpost spans. Applying
`entry + 30*linear0 + 10*full3 + terminal + sampler` within each rank, then
selecting the largest complete rank result, gives **20.631607125ms** (rank0).
Clean all40 queued wall time exceeds this by0.785261119ms (3.6666% of wall).
That difference is **not measured host overhead**, a critical-path span or a
mathematical lower bound: this extrapolates representative reduced layers at
positions128–143 into a different full40 queued regime at positions128–254,
with different weights/routing and possible overlap/gaps.

The same reduced run's useful-byte floor is179,208,992bytes/rank at512GB/s:
**0.350017563ms ideal / 1.703294813ms native operation work / 2.015464375ms
fenced wall**. Ideal/work is20.5494%; ideal/wall is17.3666%. The0.312169563ms
wall/work difference remains unresolved dispatch/synchronization/gap cost,
not a CPU-time measurement. This same-run comparison is separate from the
clean all40 useful-byte floor and isolated queued-layer engineering budget.

For terminal scale, the same-rank native rows below account for282.987us head,
22.689us norm,14.488us head reshard,23.473us padding and21.747us stable logits
copy, in addition to casts/input layout and embedding/RoPE work. Sampling adds
436.511us of rank2 operation work in that reduced run. Do **not** add these
instrumented operation sums to clean isolated-layer wall timings and report
the result as an exact full-model decomposition. The clean token-out minus
model-only difference0.482183441ms is consistent in scale with the separate
0.487714543ms sampler control, but their changing/fixed input regimes also
prevent exact additive attribution.

Rank2 has the largest combined mean operation sum: model1.266784250ms plus
sampler0.436510563ms =1.703294813ms. Model ranks range1.263118–1.269129ms;
sampler ranks0.431209–0.436703ms. The reduced sampler share is25.6274% of
rank2 operation work; it must not become a full40 percentage. Clean full40's
sampler control is **2.2772%** of queued wall time. The reduced wall/work
difference includes instrumentation, dispatch gaps and synchronization; it is
not a measured avoidable host boundary.

The separate useful-byte floor passes:1,107,522,496bytes/rank/token at512GB/s
gives ideal2.163129875ms and10% ceiling21.63129875ms. Measured21.416868244ms
achieves10.1001% under that stated analytical byte model. This does not claim
10.1001% measured physical DRAM bus utilization or use profiler padding/rereads
to inflate useful bytes.

## Native policy and selected geometry

Values below are rank2 mean **kernel work per replay**, from the token-out
window. Program hashes index `2:<hash>` in native metadata. Source-role names
for inline kernels come from frozen production dispatch, exact operand shapes
and order; native metadata itself calls them `GenericOpDeviceOperation`.

| Role / program hash | Actual operands, fidelity and geometry | us/replay |
|---|---|---:|
| LM head /17601075575282022127|BF16 L1 width-sharded `[32(1),2048]` × BFP8 DRAM width-sharded `[2048,62080]`; HiFi2, FP32 destination, approx=false; reader3, block4, M1/N32, pipeline enabled|282.987|
| Linear packed input /9232381204257685291|BF16 `[4,1,512,3136]` coefficients, BF16 input, FP32 partial outputs; inline HiFi4; accepted K512 coefficient cache|78.917|
| Active expert GU /2350790473252368865|BF16 × BFP8, LoFi; sparse active8/256, block64, eight cores; two layer calls|72.345|
| Linear output /9268757223667248116|BF16 `[2,1,512,2048]`, BF16 L1-sharded source, FP32 partials; cached HiFi4 ordered scan|35.150|
| Full-layer router /13789318958072384102|BF16 × FP32 `[2048,256]` → FP32; HiFi4/FP32 destination; eight active cores, block2, subblock1×1|25.073|
| Linear router /13930620383976449621|BF16 `[4,1,512,256]` ordered narrow32 scan, FP32 partials, inline HiFi4|23.886|
| Active expert down /6156800978488083809|BF16 L1 input × BFP8, LoFi; block4,64cores; two layer calls|23.191|
| Terminal RMSNorm /14646700877438387314|FP32 input and gamma, HiFi2; BF16 casts surround unchanged offset RMSNorm|22.689|
| State slab /2184566274953150957|FP32 `[1,8,128,128]` state/Q/K, BF16 V, FP32 beta/decay;32cores, inline HiFi4, FP32 new state/terminal output|17.381|
| Packed full-attention QKVG /16187139709719258560|BF16 L1 × BFP8 DRAM-sharded `[2048,2560]`, HiFi2|16.249|
| Paged SDPA /15074222989599085580|BF16 Q/K/V; HiFi4/FP32 destination;8×8 grid, qchunk32/kchunk256; page32 KV unchanged|14.453|

Head geometry in the run records sixteen activation shard cores,61 output
shards of `[32,1024]`, bank storage width7776, local K2048/N62080. The native
program's90-core resource union is not sixteen compute workers; activation,
reader and output grids differ. Logical vocabulary IDs and rank offsets stay
62080; storage padding does not add selectable tokens. The native compute
source is `bmm_dram_sharded_multicore.cpp`. No BFP4 or rejected LoFi projection
policy appears in this head/full/shared path.

The selected cached scan's two exact shapes,16 coefficient tiles/worker,
32768-byte CB and+28672 transient bytes/worker are recorded in execution
geometry. The router uses the narrow32 fallback rather than the slower cached
router. State-slab dispatch records separate FP32 products, ordered16-then8
addition, BF16 consumption rounding only where selected and FP32 state commit.
Its native operand/kernel identity confirms the accepted slab path executes;
the product/direct-sum helpers remain guarded fallbacks, not extra operations
claimed to execute in this supported B1 shape.

Selected BF16 residual carrier and BF16 KV are preserved. Three BF16 direct
all-reduces total26.770us and the FP32-partial direct all-reduce13.620us; native
metadata has FP32 input **and scratch** for the latter, BF16 output dtype,
FP32 accumulation, ring4/axis1/two links, Linear topology, and L1 width-sharded
output `[32,64]` over8×4 cores. The `dtype=BF16` attribute describes its output;
it does not prove FP32 partials were narrowed before the collective. Source
`CollectiveWorkspace` owns the persistent serial scratch/semaphores; native
inputs establish that scratch participates. There are no newly inserted
inter-layer collectives.

One bounded profiler-metadata limitation: the embedding gather's cached
program record16954371894629675786 retains `[64,512]→[64,2048]` from another
use of the program, while its immediately preceding captured tilize operation
11978461959571107698 produces logical `[1,1,512]`, padded `[1,32,512]`.
`TT_METAL_PROFILER_NO_CACHE_OP_INFO` is unset. Therefore this cached row cannot
prove the decode gather's dynamic row count/bytes; source and producer identify
the singleton contract. Width512→2048, two links and its17.188us timing are
unambiguous. Do not silently report64 runtime decode rows or derive a new
payload count from this reused metadata. This is a metadata scope caveat, not
an observed output or numerical-policy failure.

## Sampling, collectives and persistence

The31-op sampling trace contains the split `TopkRoutePrep`24.436us,
`TopkLargeIndices`266.071us and `TopkRouteFinish`20.537us on local BF16
`[1,1,32,62080]`, producing local top32. Native `k=32` is physical candidate
capacity, not semantic greedy k. Semantic k1/p0/temp1 remains device-side.
`SamplingDeviceOperation` costs20.957us, seed setup18.343us; token merges,
copies and seed advance are included. No `TopKDeviceOperation`, argmax op or
full-vocabulary all-gather appears in the selected component groups.

The two sampler gathers consume BF16 values and UINT16 indices, each exactly
`[1,1,32,32]` to width128, axis1/dim3/one link, with barrier present,
`using_persistent_buffers=false` and `use_all_gather_async_via_broadcast=false`.
Their combined rank2 work is26.735us. The other gather is the width512 embedding
entry, not logits. Thus no full248320-vocabulary gather is hidden in the graph.
The large-indices selector is60.95% of sampling kernel work, **not** dominant
full-model token-out work; even the whole clean sampler is only2.28% of all40.

[Final sampler timing](final/sampler_timing.json) validates all three strategies
on real terminal logits, negative unique maxima and cross-shard ties, including
lowest-ID tie behavior. Real-terminal split greedy is0.487648860ms, padded
power-of-two split0.521799590ms, force-argmax2.669194410ms. These are correct
greedy comparisons on the actual32-row terminal contract. The selected split
path is not justified by an incorrectly shaped sampled comparator. Sampled
k>1/p>0 request/seed isolation is separately covered by
[all40 tracked/Watcher replay](final/replay_all40.json): the recorded runner
uses top-k1/8/16 and top-p1.0/0.9/0.8, then changes slot1 to k4/p0.7 while
checking untouched slots. This is not a statistical distribution-quality
claim. [Final sampler acceptance](final/sampler_acceptance.json) is the tracked
greedy three-strategy control, not evidence of a sampled native capture.

The explicit persistent-output broadcast experiment is a **measured rejected
candidate**: [raw watched diagnostics](persistent_sampler_l1_watched_v3.diagnostics.json)
show3072/4096 elements per rank unchanged poison outside its own32 columns,
before trace capture. A2048-byte BF16 tile enters a4096-byte native scatter
packet with no final partial-packet flush. The exact source explanation,
reproducer and unapplied specialization are in
[broadcast rejection](sampler_persistent_broadcast_rejection.md) and
[ownership/barrier audit](sampler_persistent_audit.md). The global2048-byte
fabric-payload workaround conflicts with selected4096-byte FP32 collective
pages. Width64 padding/compaction and a narrow native one-tile packet repair
remain **untried**, with no asserted speedup. This rejection is correctness
evidence, not an excuse based on the sampler's small percentage. Stable trace
addresses and persistent `tt_out_tok` do not imply explicit sampler AG buffers
are enabled.

## Every report advice family and its disposition

Token-out/model-only reports expose the same four combined advice strings;
sampler-only has none. The table separates their actionable clauses. These
decisions do not claim a new exhaustive search at the final full40 source.

| Advice / exact affected role | Evidence-backed disposition |
|---|---|
| Put sparse GU input0 in L1 | Inherited [sparse geometry search](../multichip_decoder/sparse_geometry/search.json) measures control0.4057965ms, L1 base0.427067ms, L1/block64 0.416677ms versus selected DRAM/block64 0.3926215ms. All have prefill PCC0.999972759/min decode0.999989250. Retain DRAM input for this indexed B1 family; this is prior whole-layer evidence, not a fresh stage7 cumulative screen. Down already uses L1. |
| Increase sparse GU/down subblock above1×1 | Same inherited search: DRAM GU4/subblock2 0.406983ms, down32/subblock2 0.408286ms, down16/subblock4 0.411432ms; corresponding L1 variants0.431215/0.428810/0.432227ms. No win over the matched0.405797ms control; all pass the stated PCCs. These are measured layouts, not proof against every combined block64/grid/subblock cross-product. Further cross-products are untried. |
| Use HiFi2/HiFi4 instead of LoFi for expert accuracy | Preserve the explicitly selected BFP8/LoFi active-expert policy. Final all40 AIME prefill/teacher top1=98%,top5=100%,top100=100% pass; no measured accuracy defect requires changing fidelity. This is a generic accuracy suggestion, not evidence of a faster correct policy. A new expert fidelity frontier is outside this stage and remains unmeasured here. |
| Put FP32-router matmul input in L1 | The exact isolated real-layer3-weight control now measures DRAM baseline0.052364922ms versus L1-copy candidate0.081232396ms (**+55.128%**), including the copy and complete router consumers. Retain baseline for this tested geometry/fixture; [comparison](router_advice_comparison.md), [CSV](router_advice_comparison.csv) and v2 artifacts establish the bounded rejection. Already-L1 producers and other geometries remain untried. |
| Increase FP32-router subblock1×1 | A legal block2/four-core/per_core_N2/subblock1×2 decomposition measures0.052359470→0.052516150ms (**+0.299%, no demonstrated gain**). Combined with the L1 copy it measures0.052339894→0.054374424ms (**+3.887%**). Retain selected eight-core/per_core_N1 baseline for these controls; wider decompositions remain untried. See [comparison and exact acceptance](router_advice_comparison.md). |
| Change sensitive router HiFi4 to HiFi2 | User-bound FP32 sensitive router policy is preserved. [Stage5 decisions](../optimized_multichip_decoder/optimization_decisions.md) and [functional repair ledger](../optimized_multichip_decoder/AUTOFIX_functional.md) retain numerical-sensitive native-family rejections, but do not isolate this precise full-layer router HiFi2 proposal. Do not label it a measured losing candidate; it is an untried policy change outside this stage's authorized numerical policy. |
| Block4 looks good; consider HiFi4 for full accuracy on shared packed288/down2048 projections | Block4/reader family has [inherited singleton geometry controls](../multichip_decoder/geometry/search.json); selected HiFi2/BF16 control and phase tradeoffs are retained in [stage5 decisions](../optimized_multichip_decoder/optimization_decisions.md). Final top-k gates pass. No new shared HiFi4 screen is claimed. Generic extra-accuracy advice does not authorize replacing the selected fidelity or relabeling prior LoFi phase tradeoffs as a HiFi4 rejection. |

The three router controls use pinned real layer3 FP32 weights and six
**synthetic** BF16 input fixtures, preserving HiFi4/FP32 destination arithmetic,
exact selector, softmax and BF16 scores. Each of six v2 processes passes192
all-rank exact comparisons of logits/values/IDs/scores, with stable program
cache: separate acceptance uses tracking1/skip0/Watcher10, while clean timing
uses tracking0/skip0 with Watcher/profiler unset. The CSV links all six JSON
artifacts. Timings are medians of three500-replay nonblocking queues with no
per-replay fence/upload/readback; they include four persistent diagnostic
output copies and are not production matmul-only latencies. These synthetic
fixtures do not establish natural-activation behavior or reject an accuracy
policy. No production candidate was promoted.

The original tracked router failure remains retained. The harness held its
last trace-local score through the Python loop variable `x` after `del out`;
`del out,x` fixes ownership without a tracker exemption. All six v2 runs pass
following that harness-only repair; initial untracked timing is superseded.
Production and frozen final-validation tests are unchanged.

The new dominant terminal matmul was actually searched at fixed BFP8/HiFi2:
[reader3/block4](head_r3_b4_c16.json) improves complete terminal0.452275→0.353635ms;
reader3/block8 and16 give0.367281/0.382754ms, reader2/block8 and16
0.371851/0.417201ms. Reader3/block32 and unsplit reader1/block1 exceed physical
L1. The legal [four-piece one-reader control](head_pressure_r1_b1_s4.json)
is2.398756ms under ballast; [selected pressure control](head_pressure_r3_b4_s1.json)
is0.361358ms versus0.460104ms baseline. Those component measurements include
norm/layout, not just GEMM, and preserve exact logits/top1. The native final
head confirms this selected reader3/block4 contract reached execution.

Custom-kernel advice is not generated automatically. Actual stage7 decisions
remain [cached/narrow scans](optimization_checklist.md#completed-repair-evidence-and-final-source-qualification-boundary),
[state-slab audit](state_slab_recurrence_audit.md) and their exact/Watcher/clean
artifacts: cached K512 input/output were kept; cached router/K1024 lost; no-clear
padding consumption was poison-tested; ordered recurrence fusion passed all
states/evolving traces. These are execution-geometry changes, not permission
to restore rejected native norm/projection/TF32 recurrence or residual families.

The largest remaining indexed work is sparse expert GU and the linear input
scan, while the new head is the largest single op in a two-layer model. Further
router geometry beyond the three bounded controls above, already-L1 router
producers, head output-grid combinations, entry embedding replication,
sampler padding/native-tail repair and broader prefill geometry are untried
opportunities. Their gains are unknown. The parent's Stop Rule/handoff record
must retain them; this profile alone does not establish that every possible
optimization was tried. The final correctness acceptance is a separate
[passing artifact](final/correctness_acceptance.json), with original serving
failures retained and v3 fixes scoped to the harness. Native profile success
does not replace its tracked/Watcher gates, nonaligned context evidence or
independent stage review.
