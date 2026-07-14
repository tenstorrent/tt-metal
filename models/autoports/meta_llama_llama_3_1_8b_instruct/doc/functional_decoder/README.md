# Functional Decoder

## Provenance

Target model: `meta-llama/Llama-3.1-8B-Instruct`.

Translated forge emit:

- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_ttnn.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/consteval.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/params.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_pt.py`
- `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/main.py`

Path classification: decode only. `model_pt.py` runs a 128-token PyTorch prefill into `StaticCache`, then returns one `next_token`, one `cache_position`, and the populated `past_key_values`. `main.py` sets `NUM_TOKENS_PER_SAMPLE = 1` and passes persistent per-layer key/value caches to `ModelTTNN`. The emitted graph updates those caches in place and calls `scaled_dot_product_attention_decode`. No emitted TTNN full-sequence prefill graph is present.

## Runtime Contract

Implemented class: `models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder.FunctionalDecoder`.

`from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, cache_len=128, weight_dtype=ttnn.bfloat16)` accepts canonical HF layer keys, layer-local keys, and `model.language_model.layers.{i}` keys. It pre-transposes weights, fuses the emitted Q/V/K packed projection for layers 0-30, handles the layer-31 Q/K/V reorder, casts weights to bf16, and moves them to TTNN DRAM TILE tensors.

`decode_forward(hidden_states, *, key_cache, value_cache, cache_position, cos, sin, attention_mask, key_cache_update_idxs=None, value_cache_update_idxs=None)` consumes:

- `hidden_states`: TTNN bf16 TILE tensor shaped `[1, 1, 32, 4096]`
- `key_cache`: TTNN bf16 TILE tensor shaped `[32, 8, cache_len, 128]`
- `value_cache`: TTNN bf16 TILE tensor shaped `[32, 8, cache_len, 128]`
- `cache_position`: TTNN int32 tensor containing the shared decode cache position
- `cos`, `sin`: TTNN RoPE tensors for that decode position
- `attention_mask`: TTNN mask shaped for decode attention over `cache_len`

It returns a TTNN bf16 tensor shaped `[1, 1, 32, 4096]`. Runtime forwards contain no `torch`, `ttnn.from_torch`, or `ttnn.to_torch` calls.

`prefill_forward(...)` raises `NotImplementedError` because the forge emit did not ship a prefill graph.

## Emitted Math

- RMSNorm epsilon: `9.9999997473787516e-06`.
- Attention: input RMSNorm, fused Q/V/K matmul for layers 0-30, layer-31 Q/K/V reorder, `split_query_key_value_and_split_heads(num_heads=32, num_kv_heads=8, transpose_key=False)`, RoPE on Q and K from HF/forge scaled `model.rotary_emb.inv_freq`, `paged_update_cache` for K and V, `scaled_dot_product_attention_decode(is_causal=False, scale=1/sqrt(128))`, `nlp_concat_heads_decode`, output projection, residual add.
- MLP: post-attention RMSNorm, `gate_proj` with fused SiLU, `up_proj`, multiply, `down_proj`, residual add.
- Static compiler layout/program configs from the emit are not copied. Runtime uses bf16/TILE/DRAM defaults except for the minimal legal L1 height shard needed by decode cache update and decode head concatenation on the available 8x8 grid.

## Validation

| Check | Result |
| --- | --- |
| Runtime fallback source audit | pass |
| Prefill stub reason | pass |
| RoPE scaled inv_freq vs HF | pass |
| Decode synthetic weights, batch 32, cache 128 | PCC 0.9990842342376709 |
| Decode real weights, batch 32, cache 128 | PCC 0.9998582005500793 |

Command:

```bash
timeout 600 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -s
```

Result: `5 passed in 27.13s`.

Context contract:

```bash
python .agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct
```

The contract records `current_supported_context = 128`, matching the emitted and validated decode cache length. Full HF context at emitted batch 32 would require 16 GiB for one layer's BF16 key+value cache before allocator overhead, weights, activations, or temporary buffers, so larger cache lengths are deferred to later stages.

## Device Notes

`timeout 60 tt-smi -ls --local` failed because the `tt-smi` entrypoint could not import `tt_smi`. A direct bounded TTNN mesh smoke passed:

```bash
timeout 120 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[0])
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

## Limitations

- Prefill is intentionally stubbed; the forge emit is decode-only.
- Decode validation and current supported context are at the emitted workload batch 32 and emitted cache length 128. Larger cache-length evidence is deferred to later stages.
- This stage does not include optimized-decoder, multichip, full-model, vLLM, watcher, Tracy, or long-context performance evidence.
