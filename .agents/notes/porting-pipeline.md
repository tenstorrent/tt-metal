# HF → TTNN Porting Pipeline

A multi-stage process that takes a HuggingFace reference model and produces a vLLM-runnable TTNN model. Per-stage outputs **accumulate** in one per-model directory; nothing is moved between dirs.

## Layout

| Where | What |
|---|---|
| `models/autoports/<model_name>/` | Per-model home. Each stage adds files here. |
| `models/common/readiness_check/` | Shared **model-readiness check** — one harness reused by every ported model. Not regenerated per-model. |
| `.agents/skills/decoder-to-productized/` | Skill that automates **one stage**: ported decoder → `tt/model.py` + `tt/generator.py` + `tt/generator_vllm.py` + vLLM registration. |

## The Generator contract

Every ported model exposes a generator that satisfies `models/common/readiness_check/contract.py`. Two API levels in one file:

- **Low-level** (`prefill_forward` / `decode_forward` with caller-managed KV cache) — used by `tt/generator_vllm.py` for vLLM serving.
- **High-level** (`generate(prompt_token_ids, max_new_tokens, *, next_input=...)`) — generator-managed KV cache. Used by demos and the readiness runner.

Each model also exposes a module-level `build_generator(model_dir, mesh_device, **kwargs)` factory. The readiness runner imports it by convention from `<model_dir>/tt/generator.py`.

Reference patterns for new generators:

- `models/tt_transformers/tt/generator.py` — load-bearing kwarg shapes and method semantics for `prefill_forward_text` / `decode_forward`. Mirror these.
- `models/demos/deepseek_v3/tt/generator.py` — a full standalone Generator that doesn't depend on tt_transformers internals. Closer to what most new ports look like.
- `models/tt_transformers/tt/model.py::Transformer` and `models/tt_transformers/tt/attention.py::init_kv_cache` — structural patterns for the model wrapper and paged KV allocation.

## When porting a new model

1. Ported decoder + supporting modules under `models/autoports/<model_name>/`.
2. `<model_dir>/tt/generator.py` implementing the `GeneratorBase` contract (subclass it for documented inheritance; the runner uses the `Generator` Protocol for type checking).
3. `<model_dir>/tt/generator_vllm.py` — thin delegate for tt-transformers-family models; thick (custom prefill/decode) for DeepSeek-style. See the SKILL.md for the vLLM plugin contract.
4. Reference file generated via `python -m models.common.readiness_check.generate ...`, saved under `models/common/readiness_check/references/<model_name>.refpt`.
5. Verify via `python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/<model_name> --reference ... --mesh-device N150`.

The shared readiness check expects **single-prompt, greedy/argmax** semantics for deterministic top-K hit-rate comparison against the HF teacher.

## Glossary discipline

- The thing between stages is a **directory**, not an "optimizer." Don't introduce that term as a pipeline boundary.
- The verification harness is the **model-readiness check**, not a per-model test.
