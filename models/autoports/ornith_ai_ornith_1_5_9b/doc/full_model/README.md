# Ornith-1.5-9B full model

Full-model batch-1 performance is **92.6 ms TTFT and 111.9 decode tokens/s/user**
(8.94 ms/token), at **128 input / 128 output tokens**, configured context 2048,
on TP4 P300c Blackhole.
This is the warmed median of three serial runs, including device-traced greedy
sampling, resident feedback and caller token readback. Same-workload teacher
forcing, including traced sampling and forced host-token uploads, measures
111.7 tokens/s/user (8.95 ms/token). [Raw performance](performance_primary.json).

The full-model stage is complete. Correctness, capacity and the blinded HF
comparison pass; independent [round-1 review](stage_review_round1.md) returned
**clean-pass**, with no Required Work. No vLLM implementation or server was started.

## Accuracy and capacity

| Gate | Result | Evidence |
|---|---|---|
| AIME24 chat prefill, 100 reference tokens | top1 99%,top5 100%,top100 100% | [accepted log](logs/accepted_prefill_check.log.gz) |
| AIME24 traced teacher-forcing decode, 100 tokens | top1 98%,top5 100%,top100 100% | [delivered-source log](logs/delivered_teacher_forcing.log.gz) |
| Resumed prefill,513 tokens, starts 32/68/512 | Every start passes; both wrong-history controls fail | [result](../../readiness/prefill_resume_check.json) |
| Full 32-layer maximum context | Prefill 262144 and terminal decode 262143, finite logits,820 cached programs unchanged | [capacity](capacity_full_final.json) |
| Full 32-layer batch 32/context 512 | Mixed lengths 33..64, reordered rows, solo equivalence, inactive protection, resident feedback | [batch32](batch32_full.json) |
| Serving interface inventory | 114 rows, 0 missing; pinned plugin 042a94863ad2, vLLM0.31.0 | [contract](serving_contract.json) |

Reference generation used the original HF tokenizer chat template, 100 tokens,
`--model-dir` and exact-match cache. The checkpoint is
`ornith-ai/Ornith-1.5-9B` revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
[Command/source manifest](final_gate_commands.json) binds the final full-stack
gates to their implementation. The [delivered source manifest](delivered_source_snapshot.json)
and [import normalization proof](import_format_provenance.json) account for the
equivalent cache-helper import cleanup. [Readiness extras](readiness_extras_change.json)
records the subsequent high-level unknown-keyword compatibility fix and targeted
device probe; an all-layer teacher-forcing rerun retains 98%/100%/100% accuracy. Earlier experiments remain in the work log and
are superseded only by named accepted artifacts.

## Model and preserved decoder policy

`tt/model.py` loads the 32 text layers (24 linear-attention, 8 full-attention),
untied embedding and LM head, terminal RMSNorm and original RoPE. It loads only
requested checkpoint tensor groups. Qwen-style delta norm weights are shifted
by 1. Embedding is BF16 hidden-sharded; the BF8 LM head is vocabulary-sharded.
Text generation uses all layers; the optional [0,3] stack is diagnostic only.

The stage starts from the reviewed optimized multichip decoder at base commit
c2254f91c38840cd4069a1ecbbd3f4874e27d557, decoder SHA256
b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8.
The production strategy remains TP4 on four P300c Blackhole devices 0–3,
mesh 1x4, FABRIC_1D,shared MultichipCollectives,Linear topology,two links.
BF8 projection weights/CCL, attention HiFi2/MLP LoFi, BF16 decode activations,
FP32 recurrence, and BF8 paged K/V storage with BF16 update inputs are retained.
Original shape-dependent prefill precision and 32768/16384 chunk caps remain.
Decode residuals remain replicated across ranks but width-sharded locally in
L1; inner projections are TP-sharded. There is no inter-layer gather or reshard.
The existing [rejection ledger](../optimized_multichip_decoder/work_log.md)
remains authoritative; no rejected policy was silently restored.

Decoder edits add global physical-page pool ownership and masked-only safe
RoPE lookup. The original all-active compute path is retained. The wrapper
protects inactive recurrent/conv rows with prepared device scratch; attention
kernels use their native signed -1 inactive-position contract.

## Public generation and serving boundaries

`build_generator(model_dir,mesh_device,**kwargs)` implements the standard Metal
readiness Generator. `generate(...,enable_trace=True)` owns standalone request
setup, logical lengths, final-token selection and repeated device token feedback.
It supports a configured capacity 1..32, retaining the full native pool and
masking peers for a single high-level request. `host_sampling=True` is explicit
greedy compatibility; it is excluded from token-out performance.

`prefill_forward(tokens, page_table=...,kv_cache=...,prompt_lens=...,
start_pos=...,empty_slots=...,request_lens=...)` accepts mixed prompts, fixed
slots, arbitrary row order, nonzero starts and non-aligned ends. Input rows
contain complete known history through each absolute end. Explicit request
horizons preserve canonical precision profiles across chunks. If an implicit
horizon changes the profile, the supplied known prefix is rebuilt on device.
Unaligned resumes retain bounded 32-row layer input tails and aligned FP32
state, then recompute only the necessary overlap under that profile.
The public runtime owns physical padding, masks, RoPE positions, cache fill
and output slicing. Prefill uses prepared eager execution; `enable_trace=True`
is rejected before mutation. The supported trace mode is `decode_only`.

`decode_forward(tokens,start_pos,page_table=...,kv_cache=...,sampling_params=...,
reload_inputs=...,reload_page_table=...,reload_sampling_params=...,
reset_sampling_state=...,slot_remap=...)` implements pinned version1commands.
The only prepared decode bucket is the full configured capacity. Position -1
marks inactive rows, which retain model/RNG/history state and return token -1
(or zero logits). Full-pool remaps move resident token/position/table and
recurrent/sampler/history ownership simultaneously; global physical KV stays
in place. Commands are authoritative: explicit reloads copy supplied tables,
while steady resident decode ignores stale host token/position/table arguments.
Absent sampling parameters select writable full-vocabulary host logits for
plugin-owned processors/logprobs; supplied parameters select device sampling.

Async decode immediately queues owning host transfers for the exact submission.
Read/event and host-processing callbacks do not sample, advance state or replay.
They remain correct with two pending submissions and after orderly close.
Constructor cache injection avoids a second allocation. Explicit serving binding
routes cancellation/release callbacks to the actual caller cache and never frees
its storage. [Ownership](cache_ownership_full.json), [async/remap](serving_decode_probe_final2.json),
[prefill admission](prefill_generation.json), [global pool](external_cache.json).

Scheduler prefix caching is explicitly disabled for the documented
`recurrent_state` exclusion: KV pages alone cannot restore 24 recurrent states,
conv histories and aligned tails. This does not waive resumed-prefill correctness,
which passes through the public generator.

## Sampling and trace correctness

Selected math: `models/common/sampling/`. The alternative examined was
`tt-transformers/src/tt_transformers/modules/sampling/sampling_1d.py` in the
pinned adjacent dependency checkout. Both provide TP candidate sampling,
seed inputs and `tt_out_tok`. Common sampling fits the penalty/history/seed
contract better. Sampling1D was rejected for that contract gap; no custom
sampling mathematics was written. The [selection matrix](sampler_selection.md)
records state, trace, mesh, parameters, seeds, penalties, log-probability and layout contracts.

The generator owns a trace wrapper because the common high-level wrapper
has a broad corruptible scope and an eager seeded route. There are two model
traces (all-active/masked) and four sampling traces (mask×penalties), captured
once after all supported programs and resources are prepared. The model trace
produces sampler-ready vocabulary shards; the separate sampling trace writes
directly to the persistent model token input and advances the device seed.
Model position tensors advance inside their trace. Only exact producer logits
allocations are acknowledged as corruptible: producer replay always precedes
its consumer, and async output transfer is enqueued before any next overwrite.
No blanket capture exemption is used.

Greedy is semantically top-k 1 with tile-shaped 32 local candidates, followed by
candidate gathering and common selection. On real reduced-model logits, the
common greedy path measured 0.482 ms versus 2.537 ms forced argmax; both selected
the same host-greedy token. The forced full-vocabulary route was rejected.
[Comparison](sampler_probe3.json). The sampled path supports k<=32, top-p,
temperature, independent seeds and all three penalties. Larger/device-unsupported
request controls use the explicit host-logits boundary via the future adapter.
[Seed/penalty transitions](sampling_transition2.json) and
[physical32 scoped formatting](prefill_generation.json) pass.

Allocation tracking includes program buffers in repeated replay. Its deliberate
[positive control](trace_alloc_tracking/positive_control.log.gz) rejects an unsafe
live late allocation. Cross-request watcher coverage and the
[runtime audit](runtime_fallback_audit.md) cover host boundaries, cache ownership,
reset, trace lifetimes and the retained optimized decoder path.
The [runtime tripwire](runtime_fallback_probe.json) forbids host argmax/logits and
H2D inside replay, and matches explicit host greedy compatibility.

## Memory and supported limits

[Context contract](../context_contract.json) records measured weights, full-layer
BF8 KV, recurrence, persistent staging, sampler/helpers, shared CCL and reserved
trace memory. The maximum-context probe uses validated per-layer allocation
triples and an injected external pool. The public total-token budget requires a
configuration-matched DRAM measurement and an actually tested pool size; it never
advertises a free-memory arithmetic upper bound as proved capacity.

Context 262144 remains supported. Batch 32 is independently proved at context 512.
A 32-user pool with every request simultaneously occupying 262144 tokens needs
36.51 GB of KV per device before weights/workspace and cannot fit. Shared physical
pool sizing and the scheduler's total token budget express this resource tradeoff.
No maximum-context numerical-accuracy claim follows from the long capacity probe;
its assertions cover complete execution, finite final logits and no late programs.
Diagnostic all-token logits also require host memory proportional to length×vocab.

## Performance boundaries

The primary measurement prepares and warms the full 32-layer generator before
request timing. TTFT includes sampling admission, seed/history reset and prefill.
Each warmed run contains 127 model and 127 sampling replays, zero token/page-table
uploads, one initial position write, 128 caller token readbacks and zero captures
or releases. All 712 cached programs remain unchanged. Allocation-tracked and
watcher correctness checks are separate from performance runs. Owned CPU HF
workers were paused for the entire primary benchmark and resumed afterward.
[Exact commands and environment](final_performance_commands.json).

A separate fixed-device-input model trace measures 8.314 ms/token (120.3
replays/s), excluding sampling and readback; this is not teacher forcing. The
0.622 ms/token difference from token-out includes sampling, caller readback and
trace orchestration. The [advice-enabled reduced report](profile/decode_perf_report.txt.gz) covers one
real layer of each kind plus the real terminal path. Its exact measurement
window is retained as a lossless CSV archive and reproduces all 104 report rows.
The instrumented, layer-kind-weighted estimate is
24 × 0.243417 + 8 × 0.264020 = **7.954 ms/token** for the stack. Embedding/entry
adds 0.036 ms, terminal norm/head/masking 0.784 ms, device position advance
0.001 ms, and sampled feedback 0.605 ms. That extrapolates to 9.381 ms versus
the independently measured 8.936 ms; profiler instrumentation, merged-device
operation/gap accounting and different position windows prevent treating the
estimate as an exact full-stack timing. [Accounting](profile/layer_stack_accounting.json).

Sampling/feedback is about 6.8% of measured token-out latency and does not
dominate it. The common greedy path gathers only local candidates, with no
full-vocabulary gather. The LM-head matmul reaches 345 GB/s (67.4% of the report's
modeled bandwidth). A DRAM-sharded head configuration remains an unmeasured
optimization candidate; the report's ideal-bandwidth saving is 0.244 ms, about
2.7% of whole-model latency. This is a theoretical bound, not a measured speedup.
The prior optimized private-layer reference weights to 9.176 ms at S1024, but
includes entry/exit conversions, so it is not a strict packed-stack lower bound.
Reduced-stack wall time is never reported as full-model throughput.

The accepted HF/TT raw-story continuations were both read: coherent English,
early wording divergence, no mechanical repetition, and cap-limited endings.
[Manual review](autoregressive_review.json). The
[mechanical checker](degenerate_output.json) passes both retained autoregressive
runs. Chat-template quality coverage comes from the six shared prompts and the
matched ten-seed prompt04 control.

The fresh blind reader scored all 20 matched prompt04 samples before unblinding:
scientific overstatements occurred in HF 4/10 and TT 1/10, with no gibberish,
mechanical loops or wrong-language flags. The predefined failure threshold is TT
exceeding HF by more than 20 percentage points; observed difference is -30.
All 20 responses hit the 256-token cap without a completed final response.
These are shared model limitations, not evidence of complete answers or a broad
quality improvement. [Quality verdict](qualitative_verdict.json),
[fixed scores and unblinding](qualitative_entropy_control/unblinded_results.json),
[limitations](../eval_readiness/limitations.json). The six-prompt seed-0 suite
is screening; only the ten-seed control supports the failure-rate comparison.

## Review and checkpoint

Independent [stage review](stage_review_round1.md) returned **clean-pass** in
round 1, with no Required Work. The shared quality limitations and unmeasured
DRAM-sharded LM-head candidate remain documented next-stage considerations.
See [work log](work_log.md) for local checkpoint SHAs, exact commands and
retained failed hypotheses. No push or vLLM work is part of this stage.

Exact logs, generated text and rendered prompt files are kept as lossless gzip
archives to preserve whitespace. Raw local copies remain available;
[`archive_manifest.json`](archive_manifest.json) records their SHA256 hashes and restoration command.
