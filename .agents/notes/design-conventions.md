# Design conventions for porting-pipeline infrastructure

Conventions validated while building the readiness check + Generator contract (2026-05-26). When extending the porting pipeline or adding new shared infra, follow these unless you have a concrete reason to diverge.

## Shared contracts over per-stack code

The porting pipeline targets many model stacks (`tt_transformers`, `deepseek_v3`, `llama3_70b_galaxy`, `qwen*`, `gpt_oss`, more incoming). Surveys of existing generators show ~80% interface overlap on `prefill_forward_text` + `decode_forward`, with DeepSeek as the main outlier.

Default to a **single shared contract** that all model stacks satisfy. If a model genuinely can't fit, document the divergence and absorb it in that model's generator; don't fork the runner into per-stack code paths.

## Two-level Generator API

Every per-model generator exposes **both** API levels in one file:

- **Low-level** — `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...)` and `decode_forward(tokens, start_pos, *, page_table, kv_cache, ...)`. Caller manages KV cache and page table. Used by `generator_vllm.py` (vLLM owns KV cache).
- **High-level** — `generate(prompt_token_ids, max_new_tokens, *, next_input=None)`. Generator owns KV cache and page table internally; caller sees only token IDs in/out. Used by demos and the readiness check.

The high-level method is implemented as a thin loop over the low-level methods. No separate "adapter" file; same file serves both.

## Protocol + ABC

Every shared interface ships **both** a `Protocol` (for type checking and structural compatibility) and an `ABC` (for documented inheritance with concrete docstrings). Example: `models/common/readiness_check/contract.py::Generator` (Protocol) and `GeneratorBase` (ABC).

- Protocols don't force inheritance — useful for type hints in runners.
- ABCs catch missing methods at construction time (`TypeError` instead of late `AttributeError`).
- ABC docstrings serve as the implementer's spec.

## Factory by convention

Each per-model directory exposes a module-level `build_generator(model_dir, mesh_device, **kwargs)` function in `<model_dir>/tt/generator.py`. Runners discover it by convention (`importlib.util.spec_from_file_location`) — no manifest file, no plugin registry.

`model_dir` is kept in the signature even when currently unused, so future per-model config files can be wired in without an interface change.

## Deterministic sampling for verification

The readiness check runs **greedy / argmax** sampling for deterministic top-K hit-rate comparison against the HF teacher. Generators may support stochastic sampling, but `generate()` invoked from the readiness path must take the greedy code path. Pass `sampling_params=None` to the underlying `prefill_forward_text` / `decode_forward`.

## Reference file format

`models/common/readiness_check/schema.py` defines `readiness_v1`: one `topk_tokens[G, K]` tensor per entry covering all generated positions. K defaults to 100. **Don't introduce alternative schemas per model.** Pre-existing tt-transformers / DeepSeek `.refpt` formats are intentionally not back-compatible — regenerate references in the new schema if needed.

## Don't paper over upstream divergence with model lists

If three models need slightly different bootstrap, that's a sign the contract or shared harness is missing a knob, not a sign to add `if model_name == "...":` branches in the runner. Push the divergence into per-model `build_generator(**kwargs)`.

## Glossary

- **Model-readiness check** — the shared verification harness in `models/common/readiness_check/`. Not "a test for model X."
- **Generator contract** — the Protocol + ABC defined in `contract.py`. Not "the generator API" (ambiguous) or "the adapter interface" (wrong nesting).
- Pipeline stages take a **directory** as input; don't refer to inter-stage handoff as "an optimizer" or similar.
