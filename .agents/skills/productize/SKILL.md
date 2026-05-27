---
name: productize
description: Take an existing ported TTNN decoder under models/autoports/<model> and build the full model wrapper, generator contract, vLLM adapter, and readiness-check path needed to drive the model end-to-end.
user_invocable: true
---

# Productize

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage starts from a working decoder and turns it into a model that can be driven end-to-end on real weights! The important outcome is a full, working model with a generator.

## Your Part

Starting from `models/autoports/<model>/`, add:

```text
tt/model.py
tt/generator.py
tt/generator_vllm.py
```

Use the single decoder implementation from the latest stage as your starting point e.g. multi-chip if it exists, otherwise optimized, otherwise functional. Use the same mesh device as the decoder implementation for your work.

## When porting a new model

1. Read and understand the source decoder + supporting modules under `models/autoports/<model_name>/`.
2. `<model_dir>/tt/generator.py` implementing the `models.common.readiness_check.contract.Generator` contract.
3. `<model_dir>/tt/generator_vllm.py` — thin delegate for tt-transformers-family models; thick (custom prefill/decode) for DeepSeek-style. See the SKILL.md for the vLLM plugin contract.
4. Reference file generated via `python -m models.common.readiness_check.generate ...`, saved under `models/common/readiness_check/references/<model_name>.refpt`.
5. Verify via `python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/<model_name> --reference ...`.

The shared readiness check expects **single-prompt, greedy/argmax** semantics for deterministic top-K hit-rate comparison against the HF teacher.

## Inputs To Check

| Input | Where to find it |
|---|---|
| Model directory | `models/autoports/<model_name>/` containing the ported decoder block and supporting modules. |
| HF model id or local checkpoint | Usually in `config.py` as `HF_MODEL_ID`; prefer local checkpoint paths under `/proj_sw/user_dev/` when available. |
| Mesh or device label | Usually in `config.py` as `MESH_DEVICE`; use the same mesh shape as the decoder implementation. |
| Sampling expectations | The readiness path uses greedy/argmax behavior; vLLM may later exercise stochastic sampling. |

When running readiness checks, choose `--mesh-device` from the model's configured target and the hardware actually available. If the model target is available, use it. If not, use the largest available compatible mesh and record the mismatch.

If the model architecture is too recent for the current `transformers` install, update the local environment enough to load the HF config/reference and record that in the work log.

## Model Wrapper

Write a `tt/model.py` that creates a decoder stack and produces a full implementation of the huggingface reference model using only TTNN code. Read the HuggingFace reference model carefully as this will involve porting extra operations to TTNN, for example:

- embedding;
- final norm;
- LM head, respecting tied embeddings when the HF config uses them;
- paged KV-cache exposure;
- on-device sampling (reuse models.common.sampling if at all possible here)
- logit return path expected by the generator.

Avoid TTNN weight caching (from_torch is now fast enough).

## Generator

`tt/generator.py` is the load-bearing file. It should expose `build_generator(model_dir, mesh_device, **kwargs)` and a class that inherits from `models.common.readiness_check.contract.Generator`.

Implement both API levels:

- low-level `prefill_forward(...)` and `decode_forward(...)` with caller-managed KV cache and page table;
- high-level `generate(prompt_token_ids, max_new_tokens, *, next_input=None, **kwargs)` with generator-managed KV cache.

The high-level path should be a thin deterministic loop over the low-level methods. Readiness uses greedy/argmax behavior for top-K comparison against the HF teacher. On-device sampling is preferred when the model path supports it, but correctness of the readiness path is the priority.

The generator should own tokenizer loading, page-table setup, KV-cache reset, and any trace-side state reset needed between runs. Keep vLLM-facing low-level methods explicit enough that `generator_vllm.py` can delegate to them rather than duplicating model logic.

## vLLM Adapter

`tt/generator_vllm.py` should be a thin adapter when the generator's low-level methods already match vLLM needs, and a thicker adapter only when the model shape contract requires it.

Read:

- `tech_reports/LLMs/vLLM_integration.md`;
- `models/tt_transformers/tt/generator_vllm.py`;

Add a local registration hook or document the plugin registration needed in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`. If the serving harness is not ready, sanity-check imports and state clearly what remains deferred.

## Verification

Use the shared model-readiness check:

```bash
python -m models.common.readiness_check.generate ...
python -m models.common.readiness_check.run_prefill_check ...
python -m models.common.readiness_check.run_teacher_forcing ...
```

You will need to provide the readiness check with the correct mesh device shape to match the decoder and your model.

Reuse an existing reference file when it matches the prompt set and token lengths. Generate a new one when it does not. Report top-1, top-5, and top-100 hit rates for prefill and teacher-forcing decode. Differences between prefill and decode are debugging signal: cache, position, page table, or sampling behavior often explains them. We expect top-5 ≥ 98%, top-100 = 100%. Lower top-1 (~90%) is expected from bf8 quantization. If we're getting lower than these figures then first investigate your own new model code for bugs and if that is clean dive into the decoder and try increasing the datatype precision / compute fidelity to see if that improves accuracy. It's our job to deliver a fully working model, so if the decoder we have been given needs to be fixed then we're the ones who will fix it!

If no reference exists yet, generate one:

```bash
python -m models.common.readiness_check.generate \
  --hf-model <hf-model-id-or-local-path> \
  --prompt-len 128 \
  --gen-len 256 \
  --output models/common/readiness_check/references/<model>.refpt
```

Run both complementary readiness checks:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/<model_name> \
  --reference models/common/readiness_check/references/<model>.refpt \
  --mesh-device <N150|N300|T3K|TG>

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/<model_name> \
  --reference models/common/readiness_check/references/<model>.refpt \
  --mesh-device <N150|N300|T3K|TG>
```

## Two-level Generator API

Every per-model generator exposes **both** API levels in one file:

- **Low-level** — `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...)` and `decode_forward(tokens, start_pos, *, page_table, kv_cache, ...)`. Caller manages KV cache and page table. Used by `generator_vllm.py` (vLLM owns KV cache).
- **High-level** — `generate(prompt_token_ids, max_new_tokens, *, next_input=None)`. Generator owns KV cache and page table internally; caller sees only token IDs in/out. Used by demos and the readiness check.

The high-level method is implemented as a thin loop over the low-level methods. No separate "adapter" file; same file serves both.

## Factory by convention

Each per-model directory exposes a module-level `build_generator(model_dir, mesh_device, **kwargs)` function in `<model_dir>/tt/generator.py`. Runners discover it by convention (`importlib.util.spec_from_file_location`) — no manifest file, no plugin registry.

`model_dir` is kept in the signature even when currently unused, so future per-model config files can be wired in without an interface change.

## Preferred Outputs

We vary slightly from the standard outputs defined in `.agents/notes/output-files.md` for this final stage. Produce:

```text
models/autoports/<model>/tt/model.py
models/autoports/<model>/tt/generator.py
models/autoports/<model>/tt/generator_vllm.py
models/autoports/<model>/doc/productize/work_log.md
models/autoports/<model>/doc/productize/README.md
```

## Useful References

| Topic | Path |
|---|---|
| Generator contract | `models/common/readiness_check/contract.py` |
| Readiness reference generator | `models/common/readiness_check/generate.py` |
| Batch prefill readiness runner | `models/common/readiness_check/run_prefill_check.py` |
| Teacher-forcing readiness runner | `models/common/readiness_check/run_teacher_forcing.py` |
| tt_transformers model wrapper | `models/tt_transformers/tt/model.py::Transformer` |
| Paged KV allocation pattern | `models/tt_transformers/tt/attention.py::init_kv_cache` |
| tt_transformers generator | `models/tt_transformers/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
