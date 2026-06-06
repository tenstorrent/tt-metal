---
name: full-model
description: Build a complete TTNN autoregressive model and generator from a HuggingFace model plus one or more working TTNN block or decoder implementations. Use when assembling embeddings, block stack, final norm, logits, KV-cache handling, and generation into a full model, validating end-to-end behavior, and reporting full-model accuracy plus generator-level performance. Repo-specific file contracts belong in a project orchestration skill such as model-bringup.
---

# Full Model

This skill starts from working TTNN block or decoder code and turns it into a complete autoregressive model that can run end-to-end on real weights. The output should be a model wrapper plus a generator appropriate to the project. If a project-specific orchestration skill defines filenames, class names, readiness runners, or artifact locations, follow that contract there; this skill describes the general engineering work.

Full vLLM integration is out of scope here, but design the model and generator so a later serving adapter can drive the same low-level prefill/decode path without duplicating model logic.

## What To Build

Implement the model-specific pieces around the working block stack:

- token embeddings;
- positional handling such as RoPE, ALiBi, or model-specific position state;
- block or decoder stack, including multiple layer kinds when present;
- KV-cache allocation, reset, page-table or position handling, and repeated decode reuse;
- final norm and LM head, including tied embeddings when the HF config uses them;
- logits and on-device sampling path (use models.common.sampling for this);
- full on-device tracing for the decoder from token in all the way to token out.
- a generator that owns high-level token-in/token-out generation and exposes a clean low-level prefill/decode API for external callers.

Use the strongest correct implementation available as your block stack. If there are several candidates, choose the one with the best evidence for the target mesh and explain the choice.

## How To Approach It

Read the HuggingFace reference model and the reports for the working TTNN blocks before writing the wrapper. Understand the whole autoregressive path: input IDs, embeddings, masks, position IDs, cache layout, block calls, logits, sampling, and generated token feedback.

Keep setup-time work outside the hot runtime path: weight conversion, dtype choices, tensor layout preparation, cache construction, and tokenizer loading should not be hidden inside a measured prefill or decode forward.

Avoid hidden host fallback in a single prefill or decode pass. If host work is still needed between decode steps, make it explicit and measure its impact. The final decode path must use traced TTNN execution; eager decode is acceptable only as an intermediate bring-up path. When adding trace capture/replay or debugging trace execution failures, use `$tt-enable-tracing`.

As you add some new operations beyond the decoder modules you started with use the $multichip and $optimize and skills as necessary to match the any multi-chip sharding and keep the full model optimized. A tip: for this kind of optimization it's ok to run a version of the model that only has one layer of each kind present in the model (e.g. some models have some layers with windowed / full attention, others have some layers with mlp / moe etc). This reduces the running time. Tracy in particular has limited hardware capture buffers and can generally not capture a full model in one pass without extra calls to dump the profiler data anyway, so breaking the model up like this during optimization passes is best practice. Your final report should include a tt-perf-report these reduced-layer-count versions of the model.

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
def generate(prompt_token_ids, max_new_tokens, *, next_input=None, **kwargs): ...
```

The exact arguments can vary by model, but keep them keyword-friendly and explicit. The high-level `generate` path should be a thin deterministic loop over the low-level methods.

Make cache ownership explicit. Standalone generation often owns its cache internally; serving or other external callers may need to pass an already-allocated cache and page table. Do not bake in assumptions that prevent either mode unless the project contract intentionally does so.

## Validation

Compare full-model behavior against the HuggingFace reference with real weights. Use the project's readiness or evaluation harness when one exists. At minimum, gather evidence for:

- prefill logits or token prediction quality;
- teacher-forced decode quality across repeated positions;
- free-running autoregressive generation, with human review of coherence and failure modes;
- paged or positional KV-cache behavior;
- deterministic greedy behavior for repeated identical inputs;
- sequence lengths tested and any measured capacity limits;
- watcher, fallback, or runtime-integrity checks appropriate to the environment.

When full-model accuracy is poor, debug the new wrapper first: embeddings, final norm, LM head/tied weights, positions, masks, cache indexing, prompt lengths, page tables, and sampling all commonly fail outside the decoder itself. If the failure spans several of these boundaries and the causal chain is unclear, use `$autofix`; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix. Escalate back into decoder precision or fidelity only when evidence points there.

## Performance

Measure warmed full-model prefill and decode behavior, not just block latency. Include generator overhead, cache management, final norm, LM head, logits, and sampling in the reported numbers unless the project specifically asks for device-only timing. Use the $optimize skill to ensure on-device performance is as high as you can get it, but in addition now we care about the end-to-end time as measured from the host. Section 4.5 of `tech_reports/LLMs/llms.md` has great advice on this that you should read, understand and follow.

The figures you must report are:

- prefill time-to-first-token at a representative prompt length;
- decode tokens/sec/user over a representative generation window;
- traced decode latency;
- any material host/device gap or setup overhead discovered while measuring.

If performance regresses badly from block-level evidence, inspect data movement between embeddings, blocks, final norm, LM head, and sampling before changing precision.

## Evidence To Leave

Final full-model evidence should show:

- HF model id or local checkpoint, hardware, mesh, branch/commit, and environment;
- selected block/decoder implementation and why;
- model wrapper and generator contracts;
- state-dict mapping, tied-embedding behavior if relevant, and real-weight loading behavior;
- KV-cache, page-table or position handling, prompt lengths, and repeated decode reuse;
- full-model accuracy and qualitative generation evidence;
- determinism or repeated-run coverage appropriate to the implementation risk;
- watcher/fallback/runtime-integrity status;
- full-model prefill and decode performance;
- sequence limits and remaining risks;
- tt-perf-report output for a minmal set of the unique layers in the model.
