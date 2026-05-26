---
name: decoder-to-productized
description: >-
  One stage of the HuggingFace → TTNN porting pipeline. Takes an existing
  `models/autoports/<model_name>/` directory containing a ported TTNN decoder
  block (plus the HF reference) and builds the rest of what's needed to
  drive the model end-to-end: a full model wrapper, a generator that
  satisfies the shared contract, a vLLM adapter, and a vLLM registration.
  Verifies via the shared model-readiness check (teacher-forcing pass
  against an HF reference; vLLM serving pass is in scope but verification
  is deferred until that harness is finalised).
user_invocable: true
---

# /decoder-to-productized — Port stage: ported decoder block → readiness-verifiable model

## Background reading

Read these in full before doing anything — they are the load-bearing context for this skill:

1. [.agents/notes/porting-pipeline.md](../../notes/porting-pipeline.md) — where artifacts live, the Generator contract, the readiness check.
2. [.agents/notes/tt-transformers-gotchas.md](../../notes/tt-transformers-gotchas.md) — non-obvious requirements when interacting with tt_transformers internals (paged-attention conventions in particular). Four traps that silently break paged attention. Not documented in tt_transformers itself.
3. [.agents/notes/dev-environment.md](../../notes/dev-environment.md) — Python env, mesh devices, crash recovery, weight paths.
4. [.agents/notes/design-conventions.md](../../notes/design-conventions.md) — Protocol+ABC, shared contracts, factory by convention.

Also read a reference for what a ported decoder block tends to look like in this codebase:

5. **`models/tt_transformers/tt/decoder.py::TransformerBlock`** — typical shape: `LightweightModule` subclass, `__init__(args, mesh_device, tt_ccl, dtype, state_dict, layer_num, weight_cache_path, transformation_mats, paged_attention_config=None, use_paged_kv_cache=False, ...)`, `forward(x, current_pos, rot_mats_global=None, rot_mats_local=None, user_id=0, mode="decode", page_table=None, chunk_page_table=None, chunk_start_idx=None, kv_cache=None, batch_size=1) -> ttnn.Tensor` returning a hidden state fractured across devices. **The decoder in your input dir is the source of truth — read it and match what it actually exposes.** Don't coerce it to this signature; if it differs (extra kwargs, different return shape, no `tt_ccl`, etc.), adapt your model wrapper to whatever's there.

## When to use

The previous pipeline stages have produced a `models/autoports/<model_name>/` directory containing:

- A **ported TTNN decoder block** at `<model_dir>/tt/decoder.py` (or similarly named module). Verified independently. Its interface is whatever the decoder file itself exposes — `tt_transformers/tt/decoder.py::TransformerBlock` (Background reading §5) is a reference for the typical shape, not a required signature. Read the actual file before scaffolding.
- **Supporting modules** the decoder needs (attention, MLP, norm, RoPE), typically alongside it.
- A pin of the **HuggingFace model id or local checkpoint path** for the model being ported.

What is **not yet** there:

- A full model wrapper (embedding + decoder stack + final norm + LM head).
- A generator (KV cache + page table + prefill/decode orchestration).
- A vLLM adapter.

This skill produces those. Outputs are **added** to the same directory; nothing is moved.

## Inputs to confirm with the user before starting

| Input | Where to find it |
|---|---|
| Model directory | `models/autoports/<model_name>/` containing the ported decoder block + supporting modules. |
| HF model id or local checkpoint | Pinned in the model dir as `config.py` (module-level `HF_MODEL_ID`). If absent, ask. Prefer local checkpoint paths under `/proj_sw/user_dev/` when available — avoids HF auth. |
| Mesh / device label | `config.py` (module-level `MESH_DEVICE`); one of `N150` / `N300` / `T3K` / `TG`. |
| Sampling expectations | Implement on-device sampling (`models/common/sampling/`); the model must run argmax-equivalent sampling for the readiness path. Full stochastic sampling is exercised by vLLM later. |

If any input isn't clear, ask before scaffolding files.

## Step 1. Write `<model_dir>/tt/model.py` — full model wrapper

The decoder block is one layer; this file wraps `n_layers` copies of it into a samplable model.

### Responsibilities

| What | Detail |
|---|---|
| Construction | `__init__(args, mesh_device, dtype, state_dict, weight_cache_path, paged_attention_config, ...)`. Build embedding → `N ×` (whatever the local decoder block is called) → final norm → LM head → **on-device sampler** (see step 2's "Sampling location" note). Load weights from `state_dict`. |
| Embedding + LM head | Load from the HF state dict. Check `hf_config.tie_word_embeddings`: if True, the LM head reuses the embedding weight matrix (transpose). If False (e.g. Llama 3.1 8B), load `lm_head.weight` as a separate tensor from the state dict. |
| Decoder stack | Instantiate `n_layers` copies of the ported decoder block, passing each its `layer_num` and the matching slice of `state_dict`. Match the decoder's *actual* `__init__` signature as found in `<model_dir>/tt/decoder.py` — the `tt_transformers` `TransformerBlock` (Background reading §5) is a typical reference but your input may diverge. |
| KV cache surface | If `paged_attention_config` is set, each decoder layer's attention allocates a paged K/V via its own `init_kv_cache` (see `models/tt_transformers/tt/attention.py:382` for the canonical pattern: host shape `(max_num_blocks, n_kv_heads, block_size, head_dim)`, kept on DRAM, replicated to mesh). Expose `layer.attention.layer_past` so the generator can harvest per-layer K/V handles. |
| Forward methods | Two entry points — `ttnn_prefill_forward(tokens, *, current_pos, kv_cache, page_table, sampling_params=None, ...)` and `ttnn_decode_forward(tokens, *, current_pos, kv_cache, page_table, sampling_params=None, ...)`. Each runs the decoder stack with the right `mode` arg (`"prefill"` vs `"decode"`), applies final norm + LM head, and either returns logits (when `sampling_params is None`) or runs the on-device sampler and returns sampled tokens. Mirror `tt_transformers.tt.model.Transformer.forward`. |
| Mesh replication | Weights replicated via `ttnn.ReplicateTensorToMesh` for single-chip / small meshes; sharded for larger meshes. Cache to disk via `cache_file_name=` so re-runs are fast. |

### Key patterns to mirror

- **`models/tt_transformers/tt/model.py::Transformer`** — structural template. Reuse the construction sequence (embedding load, layer instantiation, weight caching) even when your decoder block differs.
- **`models/tt_transformers/tt/attention.py::init_kv_cache`** — exact shape, dtype, and mesh-replication pattern for paged K/V allocation. Match it; deviating leaks into `paged_fill_cache` later.
- **`models/demos/deepseek_v3/tt/`** — full standalone example of a model wrapper that doesn't inherit from tt_transformers.

### Anti-patterns

- **Don't allocate the KV cache at the top level.** Each decoder layer's attention allocates its own via `init_kv_cache` so the layout matches what `paged_fill_cache` expects (see [tt-transformers-gotchas.md §1](../../notes/tt-transformers-gotchas.md)). The generator harvests the per-layer handles after construction:
  ```python
  tt_kv_cache = [layer.attention.layer_past for layer in model.layers]
  ```
- **Don't bake `max_batch_size` / `max_seq_len` into the model** in ways that prevent vLLM passing different values later. Pass them through `args`.
- **Don't monkey-patch upstream tt_transformers** to inject the local decoder block (e.g. `models.tt_transformers.tt.model.TransformerBlock = LocalTransformerBlock`). It works but mutates the upstream module globally and breaks the moment any other code path imports the real `Transformer`. Either:
  - subclass `tt_transformers.tt.model.Transformer` and override `__init__` to instantiate `self.layers` from the local decoder block, *or*
  - write the model wrapper from scratch in `tt/model.py`, mirroring the `Transformer` structure but using your local decoder block directly. This is what step 1 expects.

## Step 2. Write `<model_dir>/tt/generator.py` — contract surface

This is the load-bearing artifact. It must satisfy the contract in `models/common/readiness_check/contract.py` (`Generator` Protocol + `GeneratorBase` ABC) and expose a module-level `build_generator(model_dir, mesh_device, **kwargs)` factory the readiness runner imports by convention.

### Contract surface (mandatory)

| Layer | Methods | Used by |
|---|---|---|
| Low-level (caller-managed KV cache) | `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, **kw)`, `decode_forward(tokens, start_pos, *, page_table, kv_cache, **kw)` | `tt/generator_vllm.py` (vLLM owns the KV cache) |
| High-level (generator-managed KV cache) | `generate(prompt_token_ids, max_new_tokens, *, next_input=None, **kw) -> list[int]` | demos, readiness check |
| Attributes / lifecycle | `tokenizer`, `reset()` | both |

Greedy / argmax sampling only on the readiness path. `generate()` returns the model's own predictions, not the forced tokens — the readiness runner uses these for accuracy comparison.

**Sampling location:** **Implement on-device sampling** in the model wrapper (`tt/model.py`). It's what vLLM serving will need shortly, and tt-metal already provides the building blocks under `models/common/sampling/` (`SamplingParams`, `tt_sampling`, `tt_log_probs`, `tt_penalties`). `models/tt_transformers/tt/generator.py` shows how `SamplingParams` is threaded through `prefill_forward_text` / `decode_forward`.

The low-level `prefill_forward` / `decode_forward` should accept a `sampling_params` kwarg and return:
- logits (shape `[batch, vocab]` for decode, `[batch, 1, vocab]` for prefill) when `sampling_params is None`
- sampled tokens (shape `[batch]` for decode, `[batch, 1]` for prefill) when `sampling_params` is provided

This matches the contract's "logits or sampled tokens" allowance and what vLLM will pass in.

For the **readiness path**, the high-level `generate()` still needs deterministic argmax for top-K hit-rate comparison. Pick one of:
- pass `SamplingParams(temperature=0, ...)` so the on-device sampler runs in argmax mode and returns tokens directly, or
- pass `sampling_params=None` and host-argmax the returned logits.

Either produces identical token IDs for greedy sampling. The first option exercises the on-device sampler in the readiness test (recommended).

### Responsibilities

`__init__` builds the model from step 1, allocates a page table, captures KV cache handles, and stores the tokenizer.

`prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, **kw)` runs `model.ttnn_prefill_forward` with the caller-supplied `kv_cache` + `page_table`, returns logits at the last prompt position.

`decode_forward(tokens, start_pos, *, page_table, kv_cache, **kw)` runs `model.ttnn_decode_forward` with caller-supplied state, returns logits at the current position. (May return `(logits, log_probs)` — be consistent across calls.)

`generate(prompt_token_ids, max_new_tokens, *, next_input)`:
1. Pack prompt as `[1, prompt_len]` int32 tensor.
2. Call `prefill_forward` with the model's KV cache + page table. Argmax the last-position logits → first prediction.
3. If `next_input` is provided, call `next_input(0, first_pred)` for the next input token; else feed the first prediction.
4. Loop `max_new_tokens - 1` times: `decode_forward`, argmax, append prediction, query `next_input`, advance `current_pos`.
5. Return the list of model predictions.

`reset()` zeros the KV cache (`ttnn.mul(k_cache, 0, output_tensor=k_cache)` per layer) and clears any cached trace-side state.

### Required scaffolding

- Module-level `build_generator(model_dir, mesh_device, **kwargs)` factory function (the readiness runner imports it by convention from `<model_dir>/tt/generator.py`).
- Class inherits `models.common.readiness_check.contract.GeneratorBase` so missing methods fail at construction.
- `tokenizer` attribute set from the HF tokenizer (load via `AutoTokenizer.from_pretrained`).
- All four gotchas in [tt-transformers-gotchas.md](../../notes/tt-transformers-gotchas.md) handled: kv_cache nesting if you build on tt_transformers utilities, `max_seq_len` power-of-2, decode_forward tuple unwrap, `HF_MODEL`+`TT_CACHE_PATH` interaction when using a local checkpoint.

### Reference patterns

- **`models/tt_transformers/tt/generator.py`** for the shape of `prefill_forward_text` / `decode_forward` — same kwargs, same return semantics. Mirror these in your `prefill_forward` / `decode_forward`.
- **`models/demos/deepseek_v3/tt/generator.py`** for a full standalone Generator that doesn't depend on tt_transformers — closer to what most new ports look like.

## Step 3. Write `<model_dir>/tt/generator_vllm.py` (produce, verification deferred)

Produce so vLLM can later consume the model. The runtime contract is in `tech_reports/LLMs/vLLM_integration.md`. Thin (~80 lines) when your generator's low-level methods match what vLLM expects; ~300 lines for thick adapters with custom shapes.

| Method | What it does |
|---|---|
| `initialize_vllm_model` | Builds the model from HF config + mesh; sets `use_paged_kv_cache=True` (vLLM will allocate KV later; model must not allocate at init time). |
| `allocate_kv_cache` | Allocates paged K/V tensors with the shapes vLLM passes in. Reuse `allocate_vllm_kv_cache` from `models/tt_transformers/tt/generator_vllm.py` if shapes match. |
| `prefill_forward` / `decode_forward` | Delegate to your `generator.py` low-level methods. |
| `warmup_model_prefill` | Trace warmup for the longest prefill. |
| `model_capabilities` | e.g. `supports_prefix_caching`, sliding-window flags. |
| `get_max_tokens_all_users` | Model + device heuristic. |

Templates: `models/tt_transformers/tt/generator_vllm.py` (thin), `models/demos/deepseek_v3/tt/generator_vllm.py` (thick).

**Do not run the vLLM serving pass yet.** That harness is being designed. Produce the file, sanity-check imports, stop there.

## Step 4. Register with vLLM (produce, verification deferred)

**Harness / test only** (no vLLM repo change):
```python
from vllm.model_executor.models.registry import ModelRegistry
ModelRegistry.register_model(
    "TT<Model>ForCausalLM",
    "models.demos.<model_name>.tt.generator_vllm:<Model>ForCausalLM",
)
```
Launch vLLM with `additional_config: { "tt": { "register_test_models": true } }`.

**Production**: add an entry to `register_tt_models()` in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`. The HF arch string gets `TT` prepended (`LlamaForCausalLM` → `TTLlamaForCausalLM`).

Default to harness-only; ask the user before adding a plugin entry.

## Step 5. Verification — teacher-forcing pass

### One-time setup: generate the HF reference

The teacher-forcing pass compares the TT model's predictions against a saved HF reference (top-K next-token candidates at every generated position; schema in `models/common/readiness_check/schema.py`).

If no reference exists yet for this model:

```bash
# Write a prompts file (JSON list). Reuse models/common/readiness_check/prompts/llama31_8b_instruct.json
# as a template — keep at least one factual prompt and one reasoning prompt.

/localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m models.common.readiness_check.generate \
    --hf-model <hf-model-id-or-local-path> \
    --prompts-file models/common/readiness_check/prompts/<model>.json \
    --max-new-tokens 128 \
    --output models/common/readiness_check/references/<model>.refpt
```

Runs on whichever HF can use (CPU or GPU). ~1.7 tok/s on the dev box's CPU for an 8B model; ~2 minutes per 128-token prompt.

### Run the readiness check

```bash
tt-smi -r   # if a previous run crashed

/localdev/tcheda/tt-metal/python_env/bin/python3 \
  -m models.common.readiness_check.run_teacher_forcing \
    --model-dir models/autoports/<model_name> \
    --reference models/common/readiness_check/references/<model>.refpt \
    --mesh-device N150
```

The runner imports `<model_dir>/tt/generator.py::build_generator` by convention. No per-model glue is needed beyond your `tt/generator.py`.

### Expected output

Per-entry and aggregate top-1 / top-5 / top-100 hit rates:

```
entry[0]             top1=1.000 (34/34)  top5=1.000 (34/34)  top100=1.000 (34/34)
entry[1]             top1=0.945 (121/128)  top5=1.000 (128/128)  top100=1.000 (128/128)
AGGREGATE            top1=0.957 (155/162)  top5=1.000 (162/162)  top100=1.000 (162/162)
```

Heuristic: **top-5 ≥ 99%** and **top-100 = 100%** indicate the port is functionally correct. Lower top-1 reflects quantization (bf8 vs bf16); meaningful but not a hard fail. Report numbers; let the user set the bar.

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

## Step 6. Verification — vLLM serving pass (DEFERRED)

Out of scope for the current iteration. The vLLM serving harness is being designed; once it lands this step will register the new `generator_vllm.py`, launch vLLM, and run a fixed benchmark.

## Key references

| Topic | Path |
|---|---|
| Decoder block interface (typical reference; the input decoder is the source of truth) | `models/tt_transformers/tt/decoder.py::TransformerBlock` |
| Generator contract (Protocol + ABC) | `models/common/readiness_check/contract.py` |
| Readiness runner (teacher forcing) | `models/common/readiness_check/run_teacher_forcing.py` |
| Reference generator (HF teacher) | `models/common/readiness_check/generate.py` |
| Reference file schema | `models/common/readiness_check/schema.py` |
| Full model wrapper — structural template | `models/tt_transformers/tt/model.py::Transformer` |
| Paged KV allocation pattern | `models/tt_transformers/tt/attention.py::init_kv_cache` |
| On-device sampling utilities | `models/common/sampling/` (`SamplingParams`, `tt_sampling`, `tt_log_probs`, `tt_penalties`) |
| Sampling threading reference | `models/tt_transformers/tt/generator.py` (how `SamplingParams` flows through prefill / decode) |
| Generator — kwarg shapes to mirror | `models/tt_transformers/tt/generator.py` |
| Generator — full standalone example | `models/demos/deepseek_v3/tt/generator.py` |
| Thin vLLM adapter template | `models/tt_transformers/tt/generator_vllm.py` |
| Thick vLLM adapter template | `models/demos/deepseek_v3/tt/generator_vllm.py` |
| Working reference run (single-prompt N150 paged) | `pytest models/tt_transformers/demo/simple_text_demo.py -k "batch-1 and performance and not accuracy" -s` |
| vLLM contract spec | `tech_reports/LLMs/vLLM_integration.md` |
| vLLM plugin registration | `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` |
| Shared vLLM KV helpers | `models/tt_transformers/tt/generator_vllm.py::allocate_vllm_kv_cache*` |
