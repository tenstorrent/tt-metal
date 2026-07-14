# Functional Decoder Work Log

## 2026-07-14

- Used `$forge-functional-decoder` for the forge-to-functional translation workflow and `$tt-device-usage` for hardware-facing checks.
- Read forge emit under `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`.
- Classified the emit as decode-only:
  - `model_pt.py`: `BATCH_SIZE = 32`, `INPUT_SEQUENCE_LENGTH = 128`; CPU prefill populates `StaticCache`, then returns `next_token`, `past_key_values`, and one `cache_position`.
  - `main.py`: `NUM_TOKENS_PER_SAMPLE = 1`; passes per-layer persistent key/value caches.
  - `model_ttnn.py`: consumes cache tensors, calls `paged_update_cache` and `scaled_dot_product_attention_decode`, returns updated caches.
- Implemented:
  - `tt/functional_decoder.py`
  - `tt/__init__.py`
  - `tests/__init__.py`
  - `tests/test_functional_decoder.py`
  - `doc/context_contract.json`
  - `doc/functional_decoder/README.md`
  - `doc/functional_decoder/work_log.md`
- First pytest attempt failed because the emitted 11-wide shard grid was illegal on the available 8x8 single-chip grid. Replaced it with a minimal legal 8x4 L1 height-sharded range for decode-only cache update/head-concat ops.
- Second pytest attempt exposed a test-reference shape bug: the HF layer return was indexed as if it were a tuple. Removed the stale tuple indexing.
- Independent rereview found that `build_decode_rope` recomputed plain theta frequencies instead of using HF/forge scaled `model.rotary_emb.inv_freq`. Fixed the helper to source `LlamaRotaryEmbedding(hf_config).inv_freq`, added a no-device RoPE comparison test against HF, and reran decode PCC evidence.

## Validation

Static checks:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
```

Runtime source audit result: no `torch`, `import torch`, `ttnn.from_torch`, or `ttnn.to_torch` in `prefill_forward`, `decode_forward`, or `forward`.

Hardware health:

```bash
timeout 60 tt-smi -ls --local
```

Result: failed before hardware query with `ModuleNotFoundError: No module named 'tt_smi'`.

```bash
timeout 120 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[0])
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: `MESH_SMOKE_OK`.

Functional tests:

```bash
timeout 600 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -s
```

Initial result before scaled-RoPE fix: `4 passed in 26.75s`.

Updated command after scaled-RoPE fix:

```bash
timeout 600 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -s
```

Result: `5 passed in 27.13s`.

PCC evidence:

- Synthetic decode: `0.9990842342376709`
- Real-weight decode: `0.9998582005500793`
- RoPE scaled inv_freq check: exact bf16 equality against `LlamaRotaryEmbedding(hf_config)` at cache position 127.

Context contract:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

Initial result: `Context contract OK for models/autoports/meta_llama_llama_3_1_8b_instruct: target=131072, supported=131072 (full HF context).`

Independent stage review rejected that as overclaiming because the measured decode cache length is 128. Updated `doc/context_contract.json` to `current_supported_context = 128` with `limiting_reason = device_dram_capacity` and capacity evidence: at emitted batch 32, a full 131072-token BF16 key+value cache for one layer is 16 GiB before allocator overhead, weights, activations, and temporary buffers.

Updated result:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

Result: `Context contract OK for models/autoports/meta_llama_llama_3_1_8b_instruct: target=131072, supported=128 (DRAM-limited).`

Pending before stage closure:

- Commit stage-owned changes only.

## Stage Review

Independent review attempts:

- Review 1 verdict: `more-work-needed`; fixed context contract overclaim by recording supported context 128 with DRAM-capacity evidence.
- Review 2 verdict: `more-work-needed`; fixed RoPE setup by sourcing HF/forge scaled `LlamaRotaryEmbedding(hf_config).inv_freq`, added exact bf16 RoPE comparison test, and reran PCC evidence.
- Review 3 verdict: `clean-pass`; no required work. Residual risk noted by reviewer: layer 31 special Q/K/V reorder is code-inspected but not separately hardware-tested, and larger `cache_len` is constructor-accepted but not validated by this stage. The context contract clearly limits current supported context to 128.

Repo checkpoint baseline before commit:

- Repo: `/localdev/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/forge-functional-rerun`
- Pre-stage HEAD: `6fb20b2434540f5fe8320fa63e5a420e16134b56`
- Stage implementation checkpoint commit: `b5feec4ad73c0b4d4ca03e43e020d314174f694f`
