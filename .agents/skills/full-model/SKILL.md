---
name: full-model
description: Build a complete TTNN autoregressive model and generator from a HuggingFace model plus one or more working TTNN block or decoder implementations. Use when assembling embeddings, block stack, final norm, logits, KV-cache handling, and generation into a full model, validating end-to-end behavior, and reporting full-model accuracy plus generator-level performance. Repo-specific file contracts belong in a project orchestration skill such as model-bringup.
---

# Full Model

This skill starts from working TTNN block or decoder code and turns it into a complete autoregressive model that can run end-to-end on real weights. The output should be a model wrapper plus a generator appropriate to the project. If a project-specific orchestration skill defines filenames, class names, readiness runners, or artifact locations, follow that contract there; this skill describes the general engineering work.

The full model stage does not implement the vLLM adapter, but it must design the model and generator so a later serving adapter can drive the same low-level prefill/decode path without duplicating model logic. Treat the generator as a serving component, not only as a demo loop: external callers may own scheduling, cache/page-table state, prompt lengths, positions, sampling parameters, fixed request slots, inactive rows, and mixed-length prompts.

## What To Build

Implement the model-specific pieces around the working block stack:

- token embeddings;
- positional handling such as RoPE, ALiBi, or model-specific position state;
- block or decoder stack, including multiple layer kinds when present;
- KV-cache allocation, reset, page-table or position handling, and repeated decode reuse;
- final norm and LM head, including tied embeddings when the HF config uses them;
- logits and a canonical split-sampling token-out path using the common sampling implementation that best fits the model;
- full on-device traced decode from token in to sampled token out, implemented as model decode trace plus a traced sampling path.
- a generator that owns high-level token-in/token-out generation and exposes a clean low-level prefill/decode API for external callers.

Use the strongest correct implementation available as your block stack. If there are several candidates, choose the one with the best evidence for the target mesh and explain the choice. Before assembling all layers, stand up a reduced 1-layer full model end to end (embedding prologue → one decoder layer → norm/LM-head/sampling epilogue → the real generator loop) and validate it on the target mesh; per-layer cost scales ~linearly, so this surfaces integration, terminal-path, and trace-lifecycle bugs in minutes before the expensive all-layer build.

## Capability And Context Contract

The full model must support the context length advertised by the HF config. Start from `models/autoports/<model>/doc/context_contract.json`, then recompute it for the full layer stack. Include all loaded weights, the full-layer KV cache, page tables, trace buffers, persistent CCL buffers, and other long-lived tensors that consume device DRAM.

Also preserve the rest of the advertised model capability contract established by earlier stages: cache/state semantics, layer kinds, mode switches, and the externally visible generation behavior. Do not reduce the advertised model capability to make bringup, tests, profiling, or serving easier. A reduction is acceptable only when a hard physical device limit prevents the advertised capability from fitting or running, such as device DRAM capacity for weights + KV/cache/state + required persistent buffers. If reduced, record the byte calculation or failed capacity probe, the largest feasible supported value, and the exact construction/serving setting that uses it.

Prompt length is a logical API input. The full model and generator must accept valid prompt lengths up to the supported context, including lengths that are not divisible by internal prefill chunk, tile, block, page, or trace sizes. If an op path needs aligned physical shapes, pad or chunk internally, mask padded tokens, and slice returned logits/output state back to the logical prompt length. Do not expose assertions such as `seq_len % chunk_size == 0` unless the HF model itself has that semantic restriction.

Batch handling is part of the capability contract. Optimize primarily for batch-1 single-user latency, but do not hard-code batch 1 into the model, generator, cache, page tables, position handling, sampling, or output formatting. Support larger batches through the same low-level API and test up to batch 32 when the target hardware, memory, and harness allow it. If batch 32 cannot run, record the largest tested batch and the hard physical limit.

## How To Approach It

Read the HuggingFace reference model and the reports for the working TTNN blocks before writing the wrapper. Understand the whole autoregressive path: input IDs, embeddings, masks, position IDs, cache layout, block calls, logits, sampling, and generated token feedback.

Keep setup-time work outside the hot runtime path: weight conversion, dtype choices, tensor layout preparation, cache construction, and tokenizer loading should not be hidden inside a measured prefill or decode forward.

Do not use repeated full-stack model runs as the normal debugging loop. After one or two expensive full-model failures, build a reduced reproducer before continuing: for example, load the same full-model wrapper with one real layer of each kind, real tensor shapes, real cache/page-table shapes, real terminal norm/LM head/sampling, and a short prompt/generation length. Use that reduced full-model path to localize wrapper, trace, cache, page-table, LM-head, or sampling bugs, then return to the all-layer model for final evidence. If the reduced reproducer cannot expose the failure, record why before spending more full-stack runs.

Avoid hidden host fallback in a single prefill or decode pass. The final decode path uses traced TTNN execution. Teacher-forcing readiness runs through traced decode. Eager decode is useful only while debugging and is not completion evidence. When adding trace capture/replay or debugging trace execution failures, use `$tt-enable-tracing`.

The full-model stage owns sampling. A full model is complete only when token-out decode uses the canonical split-sampling contract:

- Before designing the token-out path, read both common sampling implementations: `models/common/sampling/` and `models/common/modules/sampling/sampling_1d.py`. Compare their contracts against this model: state ownership, trace behavior, mesh/topology support, sampling-param/seed/logprob/penalty handling, `tt_out_tok` support, top-k/layout/padding expectations, and fit with the generator. Choose the implementation that best fits the model. Record the choice and the rejected alternative in the full-model docs. If neither common implementation fits, explain the exact missing contract before writing custom sampler code;
- the model decode trace produces sampler-ready logits without host logits readback;
- the chosen sampling implementation is traced, or the generator owns a correct trace wrapper around it;
- sampling receives `tt_out_tok` pointing at the persistent decode token input tensor, so the sampled token becomes the next decode input on device;
- current-position/RoPE position state advances coherently with token feedback on device inside the trace for fixed-step decode loops; do not refresh positions from host every token;
- page-table trace inputs are refreshed only when the page table changes, with no per-token page-table copy in the unchanged-page-table case;
- greedy decode stays on device and uses the fastest correct on-device sampling strategy for the target mesh. Force-argmax is optional. Benchmark it against the normal top-k/top-p-capable sampling path when terminal sampling is material.
- the split-sampling shape/layout contract holds: each rank's local candidate tensor is a complete tile before the candidate all-gather, and the CCL/`ttnn.sampling` innermost dim is tile-aligned. A one-tile-per-packet candidate gather and a post-gather `ttnn.pad` are known device-hang hazards; keep the sampler/gather choice open and prefer a variant that avoids them (reducing the candidate-gather payload to a bf16/multi-tile packet so it never takes the one-tile path is a first-class fix, not only an accuracy knob);

The same path supports top-k/top-p sampling. Do not complete the full model with a one-off greedy-only path and leave sampled serving to be invented in vLLM.

The delivered token-out path has no host argmax, full-logits readback, untraced sampling inside the model trace, or Python readback/writeback loop. If this contract does not work, keep the full-model stage failing and debug it there before starting vLLM.

Some shared tests require host-side sampling. Support that as an explicit compatibility mode in the generator when needed, but keep it separate from the optimized measured path. Host sampling must never replace the on-device traced sampling path used for token-out performance or vLLM serving.

Build decode around persistent device state:

- allocate stable token, current-position/RoPE, page-table, KV-cache, sampler, output, and any CCL buffers before capture;
- release captured decode/sampling traces before allocating for the next request's prefill; traces held live across a reset make the next prefill's allocations unsafe (host-side dispatch stalls), so `reset()`/request-teardown must free them first and recapture per request;
- capture model decode and sampling traces over those stable tensors;
- feed the next token through `tt_out_tok`, not a host reconstruction path;
- advance current-position/RoPE state on device for each replay when the decode step is a simple increment;
- skip page-table copies when the page table is unchanged;
- avoid per-token host mask construction, cache reset, synchronization, blocking trace replay, or feedback readback.

If TTNN or the runtime API blocks one of these items, keep the stage incomplete until it is fixed or you have the smallest repro for the blocked item. Do not treat a full-model decode result as complete while it still has avoidable host work between trace replays.

Use `$multichip` and `$optimize` for full-model pieces added around the decoder stack. Match the decoder's multi-chip sharding and keep the full model optimized. For profiling, build a reduced variant with one real layer of each kind, such as one sliding-attention layer and one full-attention layer. Keep real tensor shapes, sequence shapes, sharding, dtypes, KV-cache/page-table shapes, final norm, LM head, sampling, and trace behavior. Do not run Tracy or device-profiler collection on the full all-layer model stack. This keeps profiling fast and avoids multi-GB profiler dumps and Tracy buffer limits. Final evidence should include a `tt-perf-report` for the reduced-layer profiling variant.

The generator should expose two conceptual levels:

- low-level prefill/decode methods where a caller can manage cache, page tables, positions, prompt lengths, and batch state;
- high-level generation that owns tokenizer/setup state and loops deterministically over the low-level methods.

The standard Metal generator interface is:

```python
def build_generator(model_dir, mesh_device, **kwargs): ...
```

with a concrete generator class implementing `models.common.readiness_check.contract.Generator` when that contract exists in the checkout.

Expose both API levels with signatures along these lines:

```python
def prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...): ...
def decode_forward(tokens, start_pos, *, page_table, kv_cache, ...): ...
def generate(prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, **kwargs): ...
```

The exact arguments can vary by model, but keep them keyword-friendly and explicit. `enable_trace` must be an explicit keyword on `generate`; accepting it only through `**kwargs` is not enough because the readiness teacher-forcing runner requires traced decode. The high-level `generate` path should be a thin deterministic loop over the low-level methods.

Make cache ownership explicit. Standalone generation often owns its cache internally; serving or other external callers may need to pass an already-allocated cache and page table. Do not bake in assumptions that prevent either mode unless the project contract intentionally does so.

Preserve the decoder's inter-layer data layout unless changing it is faster end to end (and measured evidence proves it). If the optimized multichip decoder keeps the residual stream sharded/fractured across devices, do not force every layer to all-gather back to a replicated full hidden state simply because that is easier for the wrapper. Prefer the same layout contract and fused collective/matmul patterns used by the mature implementation; if a final gather is needed for norm, LM head, or sampling, localize it to the terminal path and measure it separately. Carry one decode tensor layout across the whole stack (avoid a per-layer tilize/untilize round-trip), and use causal SDPA with shared per-model RoPE tables rather than a materialized context×context mask or per-layer RoPE duplication.

## Validation

Compare full-model behavior against the HuggingFace reference with real weights. Use the project's readiness or evaluation harness when one exists. At minimum, gather evidence for:

- prefill logits or token prediction quality;
- teacher-forced decode quality across repeated positions;
- free-running autoregressive generation, with human review of coherence and failure modes;
- paged or positional KV-cache behavior;
- deterministic greedy behavior for repeated identical inputs;
- logits output by the ported model on a given prompt being deterministic across repeated runs and batch positions
- sequence lengths tested and any measured capacity limits;
- at least one valid non-aligned prompt length through the public generator path, preferably near a chunk/page boundary and one long non-divisible length near supported context;
- watcher, fallback, or runtime-integrity checks appropriate to the environment.

Run correctness from the smallest useful surface to the full gate. Start with a short smoke check on the reduced full-model probe when the all-layer gate is slow, then move to the complete all-layer model after the smoke check passes. If a test, prompt, or command fails, rerun that failing item directly while debugging; do not rerun the whole suite after every edit. Once the targeted failure is fixed, rerun the smoke check, then rerun the full all-layer correctness gate for final evidence.

Include both batch-1 and larger-batch correctness coverage. Batch 1 is the primary latency target, but the model must still handle batch dimensions correctly. Add at least one batch >1 test for prefill, decode, cache/page-table indexing, token feedback, and output formatting; test up to batch 32 when the hardware and harness allow it.

For instruction/chat models, prefer a teacher-forcing reference generated from a normal chat-template prompt over raw book text. In tt-metal autoports, use the DeepSeek AIME24 prompt set rendered by the HF tokenizer chat template as the main readiness reference, for example:

```bash
python -m models.common.readiness_check.generate \
  --hf-model <hf-model-id> \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output <model_dir>/readiness_aime24_chat.refpt
```

Raw Tale-of-Two-Cities/book references can still be useful as extra stress coverage, but they should not be the main quality gate for an instruct model unless the model lacks a usable chat template. Use `$qualitative-check` for all prompt-based quality checks, prompt-format evidence, HF controls, and raw-completion versus chat-template classification.

Generate the main reference fresh by default. Reuse a reference only when metadata under the current autoport directory proves the same HF model id and revision, tokenizer, prompt source, chat-template flag, generation length, top-k, and generation command. If any of that is missing or mismatched, regenerate the reference rather than carrying forward a possibly contaminated artifact.

Free-running comparison must be strong enough to catch feedback bugs; teacher forcing cannot see them by construction (it overrides the token-feedback path every step). Use several prompts and the longest feasible generation - at least 64-128 tokens when runtime allows - not a single short continuation. Then run:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model <hf-model-id> --missing-artifacts critical --scope autoregressive
```

and include its verdict in the stage evidence. Mechanical degeneracy - doubled tokens, single-token collapse - is a decode-loop bug, never a model property. The runner-side stage gate runs the same check.

Shift qualitative checks left: as soon as the full model can generate text, use `$qualitative-check` to run the shared qualitative prompt suite through both the HF reference and TT generator. Later stages may add serving-specific checks, but they should not be the first place these prompts are tried.

Add a focused split-sampling trace test before marking the stage complete:

- capture/replay two or more decode steps with different token and current-position values;
- assert the exact persistent trace input tensors consumed by decode changed as expected;
- assert the sampled token from step N is the token input for step N+1 without host reconstruction;
- cover unchanged and changed page-table cases;
- prove the delivered path is not rebuilding tokens, positions, RoPE indices, masks, or page tables on the host every token; if a per-token host refresh remains, the stage is incomplete;
- alternate greedy and non-greedy-capable sampling params if the generator caches trace ids by sampling mode.

Build the fast probe before any repeated debugging loop: a reduced full-model variant (one layer of each kind, short generation, real tensor/cache/page-table shapes, and the real terminal path) with a documented runtime of a couple of minutes or less. Repeating a multi-minute all-layer pass to answer single-bit questions wastes the budget the debugging loop needs; the probe pays for itself within a few iterations. This probe is for debugging only. Final correctness and performance evidence still comes from the complete all-layer model.

When full-model accuracy is poor, debug the new wrapper first: embeddings, final norm, LM head/tied weights, positions, masks, cache indexing, prompt lengths, page tables, and sampling all commonly fail outside the decoder itself. If the failure spans several of these boundaries and the causal chain is unclear, use `$autofix`; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix. Escalate back into decoder precision or fidelity only when evidence points there.

When an accuracy failure appears to follow a dtype, cache dtype, math fidelity, or CCL payload change, treat that as a precision-policy diagnosis, not as a root cause. Write down the whole active policy first: activation dtype, every weight dtype group, math fidelity per projection, cache dtype, CCL payload dtype, page/block size, cache layout, page-table policy, and decode update op. Then run controls that can distinguish a true cache/kernel defect from a policy-margin problem:

- same cache dtype with higher-precision weights/math/CCL payloads;
- same weights/math with only cache dtype changed;
- exact page/block/cache-shape probe that imports model constants instead of hard-coding page sizes, local KV heads, batch slots, or mapper behavior;
- a mature or canonical implementation that uses the same low-precision cache path on a comparable repeated-decode workload, when one exists.

"BF16 fixes it" localizes an error-budget boundary. It does not by itself prove that BFP8 cache is inherently unstable, that `paged_update_cache` is defective, or that accumulated numerical error is the root cause. State those as hypotheses until the controls above isolate them. If a reduced cache probe uses different page block size, batch width, local KV head count, update row shape, mesh mapper, or page-table policy from the full model, label it as smoke evidence only and do not use it as causal proof.

For paged decode, validate the cache/page-table allocation contract against the attention op's rounded read window, not just the requested token count rounded to page size. Record the first top-k miss index, absolute sequence position, page/block id, and dynamic attention chunk boundary before blaming dtype drift. If misses cliff at a page, tile, power-of-two, or chunk boundary, run an over-allocation control where the cache/page table covers the next SDPA decode rounded read window. A passing over-allocation control is an allocation/attention-window bug, even if BF16 happened to mask the shorter allocation.

## Performance

Measure warmed full-model prefill and decode behavior, not just block latency. Include generator overhead, cache management, final norm, LM head, logits, and sampling in the reported numbers unless the project specifically asks for device-only timing. Use the $optimize skill to ensure on-device performance is as high as you can get it, but in addition now we care about the end-to-end time as measured from the host. Section 4.5 of `tech_reports/LLMs/llms.md` has great advice on this that you should read, understand and follow.

The figures you must report are:

- prefill time-to-first-token at a representative prompt length;
- trace-verified decode tokens/sec/user over a representative generation window;
- traced decode latency;
- any material host/device gap or setup overhead discovered while measuring.

Report two decode metrics when they differ:

- traced teacher-forcing decode, which excludes sampling/token feedback and is the fair comparison to readiness/PERF-style reference numbers;
- token-out decode, which includes final norm, LM head, sampling, token feedback, readback required by the caller, and is the fair comparison to standalone generation or serving.

Be precise about the local harness. `models.common.readiness_check.run_teacher_forcing` drives `generator.generate(..., next_input=..., enable_trace=True)`: it teacher-forces the next input, but the predicted token still comes from the generator token-out path. It is therefore a token-out correctness/performance gate, not a pure logits-only decoder timing. If you need a PERF.md-style logits-only comparison, create and label a separate traced no-sampling measurement.

Do not compare a sampling-inclusive serving result against a teacher-forcing/logits-only reference without naming the boundary. If token-out decode is slower, profile terminal work separately: final norm, LM head, logits movement/all-gather, argmax/top-k/sampling trace, token readback, and trace orchestration.

Treat host steps between decode iterations as implementation bugs to remove from the steady-state path, not just performance terms to report. Add counters for trace replays, token-input refreshes, current-position/RoPE refreshes, page-table refreshes, synchronizations, and readbacks, then drive the steady-state refresh counts to zero except where the caller-visible API truly requires a readback or the scheduler changes state. If current-position/RoPE refreshes happen once per generated token, the full model still has a host-stepped decode loop; fix it with device-side state advance where possible before claiming optimized full-model performance.

Measure the primary batch-1 TTFT and trace-verified decode t/s/u with the same workload shape as the vLLM primary single-user profile (prompt 128 / generate 128 by default unless the project specifies otherwise), separate from the accuracy workload, and record the workload shape next to every number. Compare this to `vllm_benchmark.json` in later stages. The larger-batch or concurrent-serving checks prove capability and serving behavior; they are not a replacement for the batch-1 latency target.

If performance regresses badly from block-level evidence, inspect data movement between embeddings, blocks, final norm, LM head, and sampling before changing precision.

For greedy decode, the perf report must not show avoidable sampler work as the dominant token-out cost. If `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, or another sampling op dominates token-out decode, fix the LM-head/sampling contract before treating dtype, CCL, or LM-head tuning as the primary optimization target. The alternate benchmark must be a semantically equivalent greedy split-sampling path, not a slower generic sampled mode that does extra work.

Compare full-model decode against the best decoder-layer stack lower bound before declaring the result good enough:

- multiply each optimized multichip decoder-layer decode latency by that layer kind's count;
- compare the summed layer-stack ms/token with the measured full-model token-out ms/token;
- optimize the named full-model-only costs before accepting the result: final norm, LM head, sampling strategy, sampler all-gather/top-k/argmax work, trace replay, host input refresh, page-table refresh, RoPE/current-position refresh, synchronizations, token readback, and disabled CCL/persistent-buffer optimizations.

If the lower bound says the requested target is plausible but the full model misses it, the optimized-full-model pass must close that specific gap. If the lower bound itself misses the target, return to decoder/multichip optimization or a targeted precision/fidelity policy rather than spending the budget on generator code.

## Evidence To Leave

Done means all of these are true and recorded:

- HF model id or local checkpoint, hardware, mesh, branch/commit, and environment;
- selected block/decoder implementation and why;
- model wrapper and generator contracts;
- carried-forward decoder contract: weight/activation/KV/CCL dtype policy, tensor-group exceptions, and residual layout;
- state-dict mapping, tied-embedding behavior if relevant, and real-weight loading behavior;
- KV-cache, page-table or position handling, prompt lengths, and repeated decode reuse;
- context contract: HF-advertised context, full-model supported context, and any hard-physical-limit reduction evidence;
- batch contract: batch-1 primary path, largest batch tested up to 32, and any hard-physical-limit reduction evidence;
- full-model accuracy and `$qualitative-check` evidence, including prompt-format metadata, rendered prompt artifacts, HF controls, and the shared qualitative prompt suite run through the TT generator;
- split-sampling trace evidence: model trace to logits, internal sampling trace, `tt_out_tok` feedback into the persistent decode token input, current-position coherence, and page-table refresh coverage;
- determinism or repeated-run coverage appropriate to the implementation risk, including logit reproducibility across runs and batch positions;
- watcher/fallback/runtime-integrity status;
- full-model prefill plus separate teacher-forcing and token-out decode performance when those paths differ;
- layer-stack lower-bound accounting and the full-model-only costs;
- trace-loop host-work counters, including token/current-position/RoPE/page-table refreshes and sync/readback counts;
- sequence limits and remaining risks;
- tt-perf-report output for the reduced profiling variant with one real layer of each unique layer kind.
