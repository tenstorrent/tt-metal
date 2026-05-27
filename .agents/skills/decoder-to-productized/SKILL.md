---
name: decoder-to-productized
description: Take an existing ported TTNN decoder under models/autoports/<model> and build the full model wrapper, generator contract, vLLM adapter, and readiness-check path needed to drive the model end-to-end.
user_invocable: true
---

# Decoder To Productized

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage starts from a working decoder and turns it into a model that can be driven end-to-end. The important outcome is a generator that satisfies the shared readiness contract and leaves enough evidence for the next integration step.

Also read, as relevant:

- `.agents/notes/porting-pipeline.md` for the per-model layout and generator contract.
- `.agents/notes/tt-transformers-gotchas.md` before building on `tt_transformers` utilities.
- `.agents/notes/dev-environment.md` before running device checks.
- `.agents/notes/design-conventions.md` before changing shared infrastructure.

## Your Part

Starting from `models/autoports/<model>/`, add:

```text
tt/model.py
tt/generator.py
tt/generator_vllm.py
```

Do not force the decoder into a preconceived signature. Read the decoder file and adapt the wrapper to the interface it actually exposes. `models/tt_transformers/tt/decoder.py::TransformerBlock` is a useful reference shape, not a required contract.

## Model Wrapper

`tt/model.py` wraps the decoder stack into a causal LM:

- embedding;
- `n_layers` decoder blocks;
- final norm;
- LM head, respecting tied embeddings when the HF config uses them;
- paged KV-cache exposure;
- sampling/logit return path expected by the generator.

Use local TTNN patterns for weight caching, mesh placement, cache allocation, and dtype choices. If the model builds on `tt_transformers`, mirror `models/tt_transformers/tt/model.py::Transformer` and `attention.py::init_kv_cache` closely enough that paged attention behavior stays compatible.

## Generator

`tt/generator.py` is the load-bearing file. It should expose `build_generator(model_dir, mesh_device, **kwargs)` and a class that satisfies `models.common.readiness_check.contract.GeneratorBase`.

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
- `models/demos/deepseek_v3/tt/generator_vllm.py`.

Add a local registration hook or document the plugin registration needed in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`. If the serving harness is not ready, sanity-check imports and state clearly what remains deferred.

## Verification

Use the shared model-readiness check:

```bash
python -m models.common.readiness_check.generate ...
python -m models.common.readiness_check.run_prefill_check ...
python -m models.common.readiness_check.run_teacher_forcing ...
```

Reuse an existing reference file when it matches the prompt set and token lengths. Generate a new one when it does not. Report top-1, top-5, and top-100 hit rates for prefill and teacher-forcing decode. Differences between prefill and decode are debugging signal: cache, position, page table, or sampling behavior often explains them.

When a device run crashes or repeats the same impossible error after a code change, reset the device as described in `dev-environment.md`.

## Preferred Outputs

Keep durable outputs concise:

```text
models/autoports/<model>/tt/model.py
models/autoports/<model>/tt/generator.py
models/autoports/<model>/tt/generator_vllm.py
models/autoports/<model>/doc/productized/productized_bringup_log.md
models/autoports/<model>/doc/productized/productized.md
```

The work log should capture construction choices, gotchas hit, commands, crashes, and fixes. The final report should list files produced, generator contract details, readiness-check accuracy, vLLM adapter status, and remaining risks.

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
| DeepSeek standalone generator | `models/demos/deepseek_v3/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
| Thick vLLM adapter | `models/demos/deepseek_v3/tt/generator_vllm.py` |
