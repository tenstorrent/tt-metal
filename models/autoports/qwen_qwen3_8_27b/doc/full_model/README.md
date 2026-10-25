# Full autoregressive model — stage06

|Batch|Input/output per user|TTFT ms|Trace-verified token-out tokens/s/user|
|---:|---:|---:|---:|
|1|4096/128|433.812|35.186|
|32, agent target|32768/128|147101.692|7.418|
|8|4096/128|3577.570|20.978|
|16|4096/128|7316.120|13.701|
|32|4096/128|14760.040|8.822|

The B1 row measures the final native out-projection K block4 geometry
(`geometry_selection.json`). Larger-batch paths do not enter this native B1 branch.

Batch1 is the required headline: one concurrent request, all64layers, median of3
warmed end-to-end autoregressive repetitions, including sampling and caller token
readback. `benchmark_b1_perf.json` records each measurement, source hashes and installed
runtime identity; performance diagnostics are disabled. Separate tracker-on checks
include program-cache allocations. Steady-state steps have
zero host token/position/page-table copies or recaptures. Decode is28.420ms/step.
Teacher-forcing uses a different203-input/100-output AIME workload: TTFT121.608ms,
35.508tokens/s/user (`benchmark_b1_perf.json:teacher_forcing_control`). This clean control still
samples predictions before overriding the next input; it is not a logits-only timing
and is not the headline workload. Full-model device-time/roofline measurements are
unknown: stage policy permits only reduced-stack profiling.

The repeated-context generation failure is repaired and final-default quality,
native capacity and clean B1/B8/B16/B32 performance checks pass. Independent review
returned **clean-pass** (`stage_review_final.md`). Local checkpoint SHAs are recorded
in `work_log.md`. No vLLM work is included.

## Model and preserved decoder contract

`tt/model.py` and `tt/generator.py` implement the text autoregressive path for
Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` on four
Blackhole p300c ASICs, mesh1x4/ring with2links. Branch `exp26-qwen38-bringup`, base
`cdcf1091d55e793f68fd46acdbe80372337e1c34`. Checkpoint safetensor names under
`model.language_model` map to64 real decoder layers (48linear_attention,
16full_attention); untied embedding and LM head load distinct real weights. Norms
use the checkpoint's1+weight convention. Embeddings are BF16 hidden-sharded; terminal
norm/head are BF16 with a vocabulary-sharded head. RoPE uses native262144 tables.
Vision inputs are outside this text-model stage.

The accepted optimized multichip strategy is retained: BFP4/LoFi projections,
BF8 activation/CCL and full-attention KV, FP32 recurrent state, BF16 residual/norm
exceptions, inherited paged-cache dtype split and packed/separate projection choices.
Prefill carries1280 residual channels/device; decode retains inherited replication
for B1–7 and native reduce-scatter1280/channel shards for B8–32. There is no
inter-layer gather added by the wrapper. One shared CCL workspace serves all layers.
The terminal path gathers once and packs decode users into tile rows. Exact B32
logit equivalence and53.104ms→1.975ms terminal trace timing are in
`terminal_packing_b32.json`; these are host-wall terminal-only measurements.

The prior rejection ledger remains in
`../optimized_multichip_decoder/{candidate_summary.csv,topology_audit.md,work_log.md}`;
selection and runtime-policy evidence remains in that directory. Stage06 decisions
and numerical controls are recorded in `rejection_ledger.json`. No rejected
precision/layout fallback was enabled. Stage06 fixes an upstream rotary factory
bug: one core group ignored the requested FP32 destination setting, producing
slot-dependent arithmetic. Both groups now honor the existing requested HiFi4/FP32
configuration; packed execution and all decoder dtype/residual policies remain.
`AUTODEBUG_slots.md`, `AUTOFIX_slots.md`, `rope_requested_after.json` and
`rotary_build_provenance.json` record the diagnosis, real-weight regression and
rebuilt runtime. This model requires the rotary factory correction in this checkout. See `runtime_audit.md` and
`sampling_contract.md` for runtime boundaries and common sampler comparison.

## Generator and context contract

`build_generator(model_dir, mesh_device, **kwargs)` returns the standard Metal
readiness Generator. `prefill_logits` and `generate(..., enable_trace=True)` implement
the high-level contract. `prefill_forward`/`decode_forward` explicitly accept bound
cache, page table, prompt lengths/positions and fixed-slot active state. The generator
may own cache or bind caller-owned cache and a replicated device page table before
capture. Existing device maps retain their address and are read on device for cache
fill. Callers map every page read by an active request; unused pages may remain-1.
Preparation always uses a separate valid temporary map. Physical int32 page-map rows must be a multiple of8 entries (SDPA32-byte
row alignment); this is validated at binding, independently of logical prompt length. Low-level calls preserve inactive KV/recurrent cache. Masked replay still advances
physical-slot sampler seed counters and RoPE indices; inactive positions remain-1.
Schedulers that pause/resume sampled requests must explicitly restore their seed
and position state. High-level generate resets request
state, including externally bound state, by explicit contract. EOS-finished high-level
rows keep running as physical slots until peers finish; only explicit low-level
inactive slots have the preservation guarantee (see `sampling_contract.md`).

Logical prompts may have any positive length up to the configured context. The
public generator pads/chunks internally, masks tails, fills only active cache rows,
handles positions and slices output. Separate continuation calls accept inherited
128-aligned absolute starts; this does not constrain whole logical prompt lengths.
Default preparation warms every physical bucket1,32..4096, both last/all-logit terminal
paths, continuation128 variants, all-active/masked decode and sampling before capture.
An explicit `prefill_lengths` argument is reduced-probe-only partial preparation.
External preparation uses bounded scratch with the same physical tensor specifications and a valid temporary page map; original bindings are restored before trace capture. Sparse-map failure/retry and original-buffer replay pass in `external_prepare.json` and all64 B32/context33024 `external_prepare_b32.json`. Cache comparisons sample rows across all devices; they do not read the entire pool to host.

Native262144 context is preserved. All64layer B1 checks passed exact262144 prefill,
nonaligned262141 plus3generated tokens and explicit decode at262143. Finite outputs,
fixed trace IDs/program count and unchanged allocator are recorded in
`benchmark_b1_perf.json:context_validation`. These are capacity checks, not full-model HF
accuracy at native context. `../context_contract.json` separates prior per-layer
accuracy from this full-stack evidence. No batch×context Cartesian capability is
claimed; each tested pool/context is explicit. The model limit is262144; configured
pools are32768 per slot for B8/B16 and33024 for B32. Largest executed prompts are
currently4096 for B8/B16 and32768 for B32. B32's33024 allocation covers the target
32768 working set plus generation headroom rounded to256; it is not a hardware
maximum. Maximum feasible context for each larger batch has not been established.
At262144 tokens, KV alone would require36,507,222,016bytes/device for B16 and
73,014,444,032 for B32, exceeding34,178,731,008bytes/device before weights or state.

Per-device B1 native-context allocated DRAM is9,189,270,016bytes after preparation,
with24,989,460,992bytes free. `memory_capacity_plan.json` reconciles model tensor
payload, request buffers, all-layer KV/recurrent state, shared workspace and allocator
overhead. Warmup delta combines program cache, trace and persistent buffers; it is
not an isolated trace-size or transient-peak measurement. The inherited12GiB
activation/CCL and1GiB trace reserves remain labeled estimates. Agent target:
32conversations with32768-token working sets, separately recorded in
`../agentic_serving.json`.

Allocation-event audits of4096-token requests measure additional transient peaks
of226,852,864bytes/device at B1 and10,138,722,304 at B32, giving total bounds of
9,416,122,880 and28,435,986,432 respectively. Trace-release deltas are25,526,272
and118,456,320bytes/device (`benchmark_b1_perf.json`, `benchmark_b32_perf.json`).
These support the reserve checks for the measured requests; the32768-token target
generation runs separately prove execution, not a measured32768 allocation peak.

## Accuracy and generation

|Phase|Top1|Top5|Top100|Compared positions|
|---|---:|---:|---:|---:|
|Prefill|99%|100%|100%|100|
|Teacher-forced decode|99%|100%|100%|100|

Fresh AIME24 reference uses the exact HF tokenizer's normal chat template,
203prompt tokens and100greedy output tokens. Reference and metadata are
`../../readiness_aime24_chat.refpt{,.json}`. These are next-token agreement checks
from one prompt, not dataset-level AIME accuracy. `prefill_accuracy.json`,
`teacher_forcing_accuracy.json` and `full_validation_provenance.json` identify the
unmodified readiness comparators and source hashes. The repo-local autoregressive
runner preserves exact template whitespace; its two-line upstream diff and hashes
are in `readiness_preserve_prompt.patch` and `readiness_runner_provenance.json`.

HF/TT AIME completions are in `autoregressive/`. First divergence is token15: HF
“Let's parse carefully.” versus TT “Let's parse.” Both remain coherent and stop at
the100-token budget before solving. Shared explain/JSON/code prompts were run on HF
and TT; JSON is correct and stops at EOS, while the explain/code examples hit128tokens
in both controls. `qualitative_verdict.md` records direct output review and limitations;
`degenerate_output_check.log` passes. No wrong-language drift or mechanical collapse was found in those bounded AIME
and shared-suite checks. The original4096-token fixture exposed a copying loop.
Changing only native fused prefill outK48 to4 repairs it; final-default outputs
exactly reproduce the reviewed candidate on three long cases. The original prompt
now exits reasoning at887 and writes coherent prose, but is truncated at1024; HF
exits reasoning at214, so equal-budget quality parity is not claimed. Nearby4095
completes coherently atEOS892; independent4096 factual JSON is correct atEOS47.
Paired HF controls and the complete repair evidence are in `qualitative_verdict.md`.

## Trace, ownership and verification evidence

`reduced_trace_contract.json` proves device token feedback, position changes, explicit
scheduler overrides, changed/unchanged page maps, sampled/greedy reset determinism,
exact seeds including high uint32 values, EOS stopping and host compatibility.
`mixed_slots.json` covers B8 mixed prompts, nonzero inactive cache preservation on
all4devices, in-place device-map changes, external-state restoration on success and
failure, all-inactive returns and128/256 continuation controls. Trace IDs remain
stable through mixed/repeated/all-active ordering. `preparation_contract.json` covers
16boundary logical lengths plus all-logit259; final all64layer/native checks include
the added continuation preparation variants.

Canonical Sampling1D is traced separately with `tt_out_tok` pointing at the persistent
model token input. Positions/RoPE advance on device. Normal k1/p0/temp1 sampling beats
semantically equivalent force-argmax at B1/B32 and passes vocabulary-shard boundary
maxima. Host logits/argmax occur only in explicitly requested diagnostic or
`host_sampling=True` compatibility paths. Optimized token-out performance never uses
those paths. Current trace/mixed probes cover exact user-provided seeds, physical external-map
alignment, and bounded unused page-window indices at near-end odd128 continuations.
Accuracy provenance records the repaired rotary backend; each benchmark records
its Python source and loaded runtime hashes. External preparation uses a separate
scratch path and requires its own ownership/preservation evidence.

Commands, failures and repairs are recorded in `work_log.md`. Run scripts through
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m <test_module>`.
Hardware jobs are serialized. No device reset, push or vLLM integration occurred.

## Decoder-stack comparison

`layer_stack_comparison.json` compares the measured4096/128 token-out latency
with48times the accepted linear-layer latency plus16times the accepted full-layer
latency at each batch. The inputs are stage05 ordinary host trace medians, not
device timings. They include per-layer replay overhead and predate the rotary
core-group correction, so this is a performance estimate rather than exact
attribution. Full-stack capture can amortize dispatch overhead across layers.
The wrapper-specific work is separately checked through terminal packing, both
common-sampler strategies, zero steady-state input refresh counters, and reduced
profiler reports.
