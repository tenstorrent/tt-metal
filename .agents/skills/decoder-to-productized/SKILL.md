---
name: decoder-to-productized
description: Take an existing ported TTNN decoder under models/autoports/<model> and build the full model wrapper, generator contract, vLLM adapter, and readiness-check path needed to drive the model end-to-end.
user_invocable: true
---

# Decoder To Productized

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage starts from a working decoder and turns it into a model that can be driven end-to-end. The important outcome is a generator that satisfies the shared readiness contract and leaves enough evidence for the next integration step.

Also read, as relevant:

1. [.agents/notes/porting-pipeline.md](../../notes/porting-pipeline.md) — where artifacts live, the Generator contract, the readiness check.
2. [.agents/notes/tt-transformers-gotchas.md](../../notes/tt-transformers-gotchas.md) — non-obvious requirements when interacting with tt_transformers internals (paged-attention conventions in particular). Four traps that silently break paged attention. Not documented in tt_transformers itself.
3. [.agents/notes/dev-environment.md](../../notes/dev-environment.md) — Python env, mesh devices, crash recovery, weight paths.
4. [.agents/notes/design-conventions.md](../../notes/design-conventions.md) — ABC contracts, shared interfaces, factory by convention.

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

## Inputs to check starting

| Input | Where to find it |
|---|---|
| Model directory | `models/autoports/<model_name>/` containing the ported decoder block + supporting modules. |
| HF model id or local checkpoint | Pinned in the model dir as `config.py` (module-level `HF_MODEL_ID`). If absent, ask. Prefer local checkpoint paths under `/proj_sw/user_dev/` when available — avoids HF auth. |
| Mesh / device label | `config.py` (module-level `MESH_DEVICE`); one of `N150` / `N300` / `T3K` / `TG`. See mesh selection guidance below. |
| Sampling expectations | Implement on-device sampling (`models/common/sampling/`); the model must run argmax-equivalent sampling for the readiness path. Full stochastic sampling is exercised by vLLM later. |

### Mesh device selection for readiness checks

When running readiness checks, determine the appropriate `--mesh-device` based on:

1. **Model's design target**: Check `config.py` for `MESH_DEVICE` — this is what the model was configured for.
2. **Available hardware**: Query `ttnn.get_num_devices()` to see available device count:
   - 1 device → can run N150
   - 2 devices → can run N150 or N300
   - 8 devices → can run N150, N300, or T3K
   - 32 devices → can run any mesh shape

**Decision logic**:
- If model's `MESH_DEVICE` is available, use that (tests the model's intended configuration)
- If model's `MESH_DEVICE` isn't available, use the largest available mesh that fits (N150 as fallback)
- For single-chip models (N150) on multi-chip hardware, test on N150 to match the model's design

Example: T3K model on 8-device hardware → use `--mesh-device T3K`; N150 model on same hardware → use `--mesh-device N150`.

### Dependency updates

If the model architecture is too recent for the current `transformers` version (e.g., model not found in `AutoConfig`), it's fine to update the library:

```bash
/localdev/tcheda/tt-metal/python_env/bin/pip install --upgrade transformers
```

Same applies to other HF dependencies (`tokenizers`, `accelerate`, etc.) if needed for the specific model being ported.

## Generator

`tt/generator.py` is the load-bearing file. It should expose `build_generator(model_dir, mesh_device, **kwargs)` and a class that inherits from `models.common.readiness_check.contract.Generator`.

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

### One-time setup: generate the HF reference

The readiness checks compare the TT model's predictions against a saved HF reference (top-K next-token candidates at every position; schema in `models/common/readiness_check/schema.py`).

If no reference exists yet for this model:

```bash
/localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m models.common.readiness_check.generate \
    --hf-model <hf-model-id-or-local-path> \
    --prompt-len 128 \
    --gen-len 256 \
    --output models/common/readiness_check/references/<model>.refpt
```

Uses batch prefill on book text. Fast (~1 second with accelerator, ~10-30 seconds on CPU).

### Run the readiness checks

Two complementary tests validate the model:

**1. Batch prefill check (fast - ~2 seconds):**
```bash
tt-smi -r   # reset cards - good practice before first run (unknown machine state) or after crashes

/localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m models.common.readiness_check.run_prefill_check \
    --model-dir models/autoports/<model_name> \
    --reference models/common/readiness_check/references/<model>.refpt \
    --mesh-device <N150|N300|T3K|TG>
```

Tests prefill accuracy via `prefill_forward(return_all_logits=True)`. Good for rapid iteration. Use mesh selection guidance above to determine the right `--mesh-device`.

**2. Decode with teacher forcing (thorough - ~30 seconds):**
```bash
tt-smi -r   # reset cards before running

/localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m models.common.readiness_check.run_teacher_forcing \
    --model-dir models/autoports/<model_name> \
    --reference models/common/readiness_check/references/<model>.refpt \
    --mesh-device <N150|N300|T3K|TG>
```

Tests token-by-token decode via `generator.generate()`. Required for release.

Both runners import `<model_dir>/tt/generator.py::build_generator` by convention. No per-model glue needed.

### Expected output

Both tests should show similar per-entry and aggregate top-1 / top-5 / top-100 hit rates:

```
entry[0]             top1=0.850 (217/256)  top5=0.945 (242/256)  top100=0.980 (251/256)
AGGREGATE            top1=0.850 (217/256)  top5=0.945 (242/256)  top100=0.980 (251/256)
```

Heuristic: **top-5 ≥ 93%** and **top-100 ≥ 97%** indicate the port is functionally correct. Lower top-1 reflects quantization (bf8 vs bf16); meaningful but not a hard fail. If accuracy differs significantly between tests, investigate:
- Prefill higher: Decode path has issues (KV cache, position encoding)
- Decode higher: Prefill path has issues (unlikely)

### Common failure modes

| Symptom | Likely cause | Where |
|---|---|---|
| `paged_fill_cache` shape error (`ShapeBase[] index out of range`) | kv_cache nesting wrong | [gotchas §1](../../notes/tt-transformers-gotchas.md) |
| `capped_warmup_seq_len must be a power of 2` | `max_seq_len` not rounded up | [gotchas §2](../../notes/tt-transformers-gotchas.md) |
| `tuple indices must be integers or slices` | forgot to unwrap `(logits, log_probs)` from decode | [gotchas §3](../../notes/tt-transformers-gotchas.md) |
| `Permission denied` writing tensor cache | `TT_CACHE_PATH` collision with absolute `HF_MODEL` | [gotchas §4](../../notes/tt-transformers-gotchas.md) |
| Repeated identical error after a code change | Device stuck | `tt-smi -r` ([dev-environment](../../notes/dev-environment.md)) |
| Lost debug output from inserted `sys.stderr.write` during pytest | pytest capture | pass `-s` ([dev-environment](../../notes/dev-environment.md)) |

When the runner crashes, save the full log and grep `Error|Traceback|TT_FATAL|TT_THROW|^info:` — the line above `Traceback` is usually the real error; the C++ backtrace is noise.

## Key references

| Topic | Path |
|---|---|
| Decoder block interface (typical reference; the input decoder is the source of truth) | `models/tt_transformers/tt/decoder.py::TransformerBlock` |
| Generator contract (ABC) | `models/common/readiness_check/contract.py` |
| Readiness runner (batch prefill) | `models/common/readiness_check/run_prefill_check.py` |
| Readiness runner (decode with teacher forcing) | `models/common/readiness_check/run_teacher_forcing.py` |
| Reference generator (batch prefill on book text) | `models/common/readiness_check/generate.py` |
| Reference file schema | `models/common/readiness_check/schema.py` |
| Full model wrapper — structural template | `models/tt_transformers/tt/model.py::Transformer` |
| Paged KV allocation pattern | `models/tt_transformers/tt/attention.py::init_kv_cache` |
| tt_transformers generator | `models/tt_transformers/tt/generator.py` |
| DeepSeek standalone generator | `models/demos/deepseek_v3/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
| Thick vLLM adapter | `models/demos/deepseek_v3/tt/generator_vllm.py` |
