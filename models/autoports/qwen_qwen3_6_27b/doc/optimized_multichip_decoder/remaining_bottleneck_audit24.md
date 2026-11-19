# Remaining material costs after profiles20 and whole-layer22

This is a source/artifact audit, not a stage-review verdict or an optimization completion claim. No TTNN import, build, device execution, or main/helper edit was performed. Prediction/composed columns, joint-output BF16, attention gate, precise native recurrence, and traced prefill are already active or queued work and are not proposed again as new experiments.

**The strongest untried existing-op family is native fused distributed RMSNorm followed by the required hidden activation gather and canonical projection input.** The lowest-risk distinct GDN fusion opportunity is correction combine → QKV selection/formatting → convolution/history update. Neither alone plausibly closes the full gap to70%; the current large matmuls are already close to their bandwidth limit, so routine grid sweeps are unlikely to do so either.

## Measured costs and the actual roofline gap

Sources are actual TP4 real-text B32/S33 reports under `perf/fusions20_{control,fused}_l{0,3}_b32_s33/`, their `accounting.json` and advice-enabled `decode_perf_report.txt`, and `fusions20_profile_comparison.csv`. These are instrumented profiles, not final uninstrumented numbers.

| Fused candidate, B32/S33 | GDN L0 | Full attention L3 |
|---|---:|---:|
| Kernel sum, us |557.491|400.2485|
| Kernel + gaps, us |574.833|419.0315|
| Same-run host decode, us |606.589|444.501|
| Read-once matrix bytes/rank |118,374,400|98,877,440|
| Mandatory recurrent RW / visible KV read bytes/rank |25,165,824 /0|0 /1,114,112|
| Existing whole-layer lower bound, us |280.352|195.296|
| Bound / same-run host |46.22%|43.94%|
| Host target at70%, us |400.503|278.994|
| Reduction needed from this profiled host, us |206.086|165.507|

Do not replace the whole-layer ratio with the report's individual matmul DRAM percentages or FLOPs percentages above100%. Do not inflate the compulsory-traffic bound with avoidable formatting, copies, duplicate reads or all idle resident matrices. New policies changing actual compulsory traffic need a new bound. Instrumented and uninstrumented ratios are separate evidence.

| Decode family | GDN profile evidence | Attention profile evidence | Interpretation |
|---|---|---|---|
| Four matmuls |261.221us total: joint input85.730, output~21, packed MLP~100, down~54|219.383us: packed input~44, output~21, packed MLP~101, down~54|Advice says all four optimized; modeled DRAM utilization about72–87%. The weight-only ideal is231.2us GDN /193.1us attention at512GB/s, leaving only~30/~26us total matrix headroom even before activation traffic.|
| Recurrent update/readout |~125us,110cores,~21.9% kernel+gap|N/A|Largest single non-matrix cost. Current prediction/native precision/column work addresses this; do not label it untried. BF16 state read/write lower bound alone is49.152us/rank, and actual SFPU work/reader scheduling consume the rest.|
| Two AG + two RS |64.513us total|~65us total|Linear async and persistent buffers already selected. Hidden activation gathers remain required inside each layer; no layer-boundary gather is required.|
| Hidden norm / staging |Two AG-following norm pairs, roughly14us each, plus sharding|Two~9us norm kernels plus conversion/sharding|Current hidden-norm fusion does not fuse the fabric stats path; native fused distributed alternatives below are different implementations.|
| GDN projection-consumer prologue |~9us correction combine +~2us QKV slice +3.153us untilize +~18us local conv|N/A|A still-material~32us four-launch sequence. Generic-op attribution follows dtype/core counts and source order, not a claim that generic opcode alone identifies the kernel.|
| Attention local processing |N/A|QK/RoPE~20us + small staging~8us; cache~5us; SDPA~13us; concat~8us; gate~4us|Packed QK/RoPE and attention-gate work are active. At long context SDPA/KV become more important; do not extrapolate S33 cost.|

At current measured matmul costs, full attention matmuls plus explicit CCL already total roughly284us, exceeding its279us70%-target before any norm/attention/cache work. Reaching70% therefore requires meaningful movement/launch overlap or savings across several families, not merely one more epilogue. GDN's matrices+recurrence+CCL total roughly451us, also above its400.5us target. This is a prioritization argument, not a claim that the stop rule is impossible.

Prefill also remains material: fused GDN profile has625.607us of chunk GDN,978.104us joint projection,548.528us packed MLP,505.787us fused MMRS,458.019us explicit AG,390.139us typecasts and329.277us reshape work. Attention has534.199us reshapes,485.220us fused MMRS,452.146us AG,386.996us packed MLP and208.822us slices. GDN first signposted kernel+gap7.514ms exceeds same-run host mean7.057ms; the accounting explicitly explains different selected-call statistics and instrumentation. Traced-prefill work is already queued and must retain fresh-state-per-call semantics.

## Latest whole-layer evidence: small gains are real but insufficient

Read actual job JSONs, not manifests as evidence. At the final read all25 `gdn22` rows and12 `attention22` rows had passing reports. The last attention report appeared while this audit was reading; none of these rows is a selected default merely because it passed.

| Fixture / decode us | Control | Column1 | Column2 | Hidden dispatch | Column1 + dispatch |
|---|---:|---:|---:|---:|---:|
| GDN regression B1S512 |484.161|466.628|473.483|480.074|462.858|
| GDN text B1S512 |484.227|466.776|473.537|480.133|463.139|
| GDN regression B32S33 |578.870|575.519|575.830|579.433|575.556|
| GDN text B32S33 |578.594|575.150|575.393|578.923|575.207|
| GDN regression B32S513 |578.685|575.418|575.593|579.301|575.354|

Column1 is meaningful at B1 (~3.6% alone); B32 gains are only~0.6%. Hidden dispatch helps B1 by~4us but does not help B32. B1 prefill differences are inconsistent across fixtures: control/column1 regression2.002/2.078ms and text2.099/2.013ms. Keep the requested repeated/default-path prefill validation rather than attributing unrelated prefill differences to a decode-only change.

| Attention fixture / decode us | Fused-QK/RoPE control | Packed | Packed + hidden dispatch |
|---|---:|---:|---:|
| Regression B1S512 |411.243|412.949|417.814|
| Text B32S33 |419.019|417.796|418.728|
| Regression B32S513 |446.391|445.712|449.272|
| Regression B32S33 |419.471|417.983|418.976|

The standalone packed-boundary win does not translate to a large whole-layer gain; B1 regresses and B32 saves only~1us. Do not repeat this same family as if its standalone speedup were still untested. Reports include `pass=true` and uninstrumented serial trace timing metadata.

## Untried native family: fused distributed hidden normalization

No invocation of `fused_rms_minimal`, `rms_allgather` or `dit_fused_distributed_rmsnorm` was found in this autoport's Python/Markdown candidate corpus, including `candidate_search.md`. Existing `family1/2`, `ag_adapted9` and hidden-norm fusions are not this fused stats-communication implementation.

Two source-backed endpoints exist:

1. `ttnn.fused_rms_minimal`, implemented under `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/`. Its default topology is Linear; the factory explicitly handles forward/backward line distances. It accepts physical32-row decode, rank4 logical `[1,1,M<=32,N]`, tile-aligned local N, width-sharded L1 input/output, row-major affine weight, matching global semaphore grid, and a caller-owned stats buffer. FP32 accumulation requires **FP32 stats**; old BF16-stats examples must not be copied under a different accumulator config. Blackhole DRAM input is explicitly unsupported; legalize once to L1 before the candidate boundary and include that conversion in its timing. Nlocal1280 has40 tiles, allowing an8-core local-width plan with5tiles/core; output must match the actual operation's local normalized shape, not be falsely labeled full5120. Verify stats geometry against the factory before allocating.
2. `ttnn.experimental.dit_fused_distributed_rmsnorm`, under its own experimental CCL directory. It fuses local statistics, fabric stats gather and post-norm; supports BF16/FP32 tiled input, logical rank4 `[1,batch,N,H]`, static affine weight, explicit Linear topology, mesh-coherent persistent stats scratch, and optional row capacity. Use `num_heads_per_device=1`, `per_head_norm=False`, no RoPE, no bias and model epsilon. Factory has Linear support and the same-repo test suite explicitly covers1×TP line meshes and5120 full width. It provides `dit_fused_distributed_rmsnorm_create_stats_buffer`; buffer geometry must match links and row capacity. The returned `persistent_output_buffer` parameter is **stats scratch**, not a reusable output tensor.

Relevant same-repo sources/tests:

- `models/tt_dit/tests/unit/test_distributed_rmsnorm_fused.py`: actual fused path, line topology, cache/owner rebinding and trace checks. Its simulated predecessor `tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_fused_rmsnorm.py` concatenates stats on one device and is not TP4 execution evidence.
- `tests/ttnn/unit_tests/operations/ccl/fusion_subtests/rms_test.py`: minimal RMS sharding, stats and output recipes; reconcile dtype and current validation rather than copy parameters blindly.
- `models/demos/gemma4_31b_qb2/tt/decoder.py` and `models/demos/deepseek_v3_d_p/tt/tt_distributed_rms_norm.py`: in-repo model uses, not Qwen correctness evidence.

**Both operations gather statistics, not normalized hidden activations.** Their output is still a local hidden shard. The complete trial must include its following hidden activation AG and any canonical matmul-input sharding, or directly feed a legally compatible next projection. Compare against current gather-first norm plus canonical matmul-input conversion, through actual projection output if feasible. Preserve the original sharded residual tensor beside this private projection activation. Returning a local normalized shard and omitting the required activation gather is invalid performance evidence; restoring an old replicated residual at the layer boundary is unnecessary.

Smallest discriminator: captured real norm input/weight from each meaningful kind at B1/B32, current exact gather-first chain versus minimal-L1 fused-stats and DiT fused-stats chains, all ending at the same width-sharded projection input and followed by the canonical packed projection. First compare intermediate output/each user and contribution through projection at unchanged>=.995; explicitly preserve FP32 norm weights and BF16 activation boundaries. Distributed reduction trees differ, so exactness must not be assumed. Do not insert host reduction or silently quantize norm weights. For nonaligned lengths, internal padding/slicing belongs to the adapter.

Warm all CCL/norm/reshape/projection branches and preallocate all retained outputs/stats/semaphores before first capture. The DiT test uses two distinct semaphore+stats sets because terminal semaphore resets can race the next invocation under rank skew; preserve that ownership protocol rather than reusing a single scratch because one synchronized call passes. Runtime bindings must rebind inputs, output, gamma, scratch and semaphores. Exercise changed-input50-queued replay, tracker1/skip0 with its live-buffer positive control, and watcher before whole-boundary uninstrumented timing. If the native entry rejects a layout, adapt its1280-width sharding/stats/padding; a first API failure is not a rejection of this material family.

This family can remove several separately scheduled norm/statistics stages, particularly in prefill; it may still lose in decode because it retains a second communication stage. Measure it before deciding. Native applicability is source-supported, not a claimed speedup.

## Distinct GDN follow-up: correction-aware convolution prologue

Current `_gdn` calls packed projection/combine, slices the first2560 QKV channels, and `_local_batched_conv` calls `recurrent_next_candidate.local_conv`, which explicitly untilizes before its96-worker B32 kernel. `packed_correction_candidate._combine` already owns direct FP32 primary/correction addition and BF16 output rounding, but has no history/taps/conv consumer. Existing `combined_conv_joint_*` rows select the two separate helpers together; they do not fuse these launches. Current source and profiles still show the intermediate operations.

A new, bounded endpoint could read the joint projection output once, apply the exact correction mapping and BF16 rounding, then perform the existing four-tap FP32 accumulation, BF16 boundary, SiLU and raw-history update for QKV. It also emits the unchanged corrected Z/B/A packed fields required by the existing beta/decay and output-gate paths. Return the same row-major BF16 post-conv `[B,1,2560]`, same packed ancillary contract, and same caller-owned history; no recurrent-state layout or interlayer contract changes. Allocate outputs before capture.

A channel-owned schedule avoids the race in a tempting convolution-inside-recurrent-reader design: recurrence column/head owners duplicate Q/K consumption, so letting each mutate shared old history would create read-after-write races. The prologue should have one writer per `(user,channel)` and read that channel's three old samples before its history store. Do not merge correction rounding into convolution multiplication or replace separate FP32 adds with FMA. Preserve BF16 raw token in history, not post-conv/SiLU data.

First compare this **complete~32us boundary**, with exact post-conv, every ancillary field and history, at B1/3/32, zero/scaled inputs, changed history, repeated updates and rebinding. Then whole-layer timing on the actual chosen joint-output dtype. The queued joint-BF16 experiment remains independent: start from its passing dtype result or the current FP32 output, and do not repeat the precision sweep. A plausible10–20us saving would be useful (~2–3% GDN B32) but is an estimate, not evidence, and cannot alone close the70% gap. No implementation is requested or supplied by this audit.

## Families not to repeat without new causal evidence

- Separate/packed MLP, reader counts, fidelity/BF8 activations: `mlp9`, `precision_geometry5`, `readers2`, and `geometry13_full_down`; dominant current matmuls already near bandwidth limits.
- Larger down K34: four `geometry15` physical-L1 adaptations establish rejection; do not call the original allocation exception the full evidence.
- Conventional/minimal fused AGMM and distributed norm+AGMM: all32 `ag_adapted9` coherent real-text boundaries passed and lost after shape/layout/native Linear repair.
- Existing decode MMRS/collective placement/persistent buffers: `family3`, `persistent_fix1`, `mmrs2`; new native fused-stats norm is not a claim that those were untried.
- Head-major recurrent→norm/gate:258 exact wide cases and complete-chain regressions in `head_major_perf19`; an immediate restore was not inserted.
- Tiled batch-preserving prefill and temporal alignment/chunking: coherent family measured,28jobs/376configs plus adaptations; do not revive absent old manifest rows.
- BFP4 KV: executed on the actual dtype; K/V reference packing floor~.977 fails. It cannot be justified by whole-output-only PCC or context-memory arguments.
- Hidden dispatch and packed QK/RoPE: latest whole-layer22 results above; neither offers an unmeasured large decode improvement.

A final profile should be collected after selected combinations settle. The present recommendations do not change the strict70%-and-two-small-kept-gains or8h occupied-device stop rule, and do not replace final default-path quality, context, ownership, fallback or review gates.
