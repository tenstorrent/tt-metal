# Functional Decoder

Target: `meta-llama/Llama-3.1-8B-Instruct`

Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Forge source: `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`

## Runtime Contract

`tt/functional_decoder.py` exposes `FunctionalDecoder(LightweightModule)` with:

- `from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, max_seq_len=128, ...)`
- `prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None)`
- `decode_forward(hidden_states, *, key_cache, value_cache, update_idxs_tensor, position_cos, position_sin, attention_mask)`

The runtime tensor convention is `hidden_states` shaped `[1, batch, seq, 4096]`. Prefill accepts arbitrary setup `max_seq_len` and batch, defaulting to the emitted batch 32. Decode is the emitted batch-32 single-token path: `hidden_states` is `[1, 32, 1, 4096]`, KV caches are `[32, 8, 128, 128]` in the emitted workload, and the decode mask is `[1, 1, 32, 128]`.

Runtime forwards contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback. Torch conversion, RoPE table construction, and mask/cache setup are restricted to `from_state_dict` and test/setup helpers.

## Emit Provenance

Translated files:

- `model_ttnn.py`: decoder block, attention, MLP, decode cache update, SDPA decode, concat-heads decode.
- `consteval.py`: emitted batch, weight layout, QKV fusion, rotary inv-freq handling, cache/mask constants.
- `model_pt.py`: HF model id, emitted batch size 32, decode workload shape, StaticCache length 128.
- `params.py`: canonical HF state-dict key mapping and absence of Q/K norm weights.

Decode graph grep result: present. The emit contains `ttnn.experimental.paged_update_cache`, `ttnn.transformer.scaled_dot_product_attention_decode`, and `ttnn.experimental.nlp_concat_heads_decode`.

Important emit details:

- The actual Llama emit has no per-head `q_norm` or `k_norm`; only input and post-attention RMSNorm are present.
- RMSNorm epsilon is HF Llama `1e-5`, matching `model_ttnn.py`.
- The actual fused QKV order differs from the prompt summary: layers 0-30 effectively use `[Q,V,K]`; layer 31 uses `[Q,K,V]`. The implementation follows the emitted graph.
- RoPE uses `ttnn.experimental.rotary_embedding`.
- Attention scale is `1 / sqrt(128)`.
- MLP is SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

## Validation

| Check | Command | PCC |
| --- | --- | --- |
| Synthetic prefill, seq 16 | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer_at_emitted_batch --tb=short -x` | 0.9984961177648134 |
| Synthetic prefill, seq 17 | same parametrized command | 0.9984850444029093 |
| Synthetic prefill, seq 64 | same parametrized command | 0.9981846012611152 |
| Real-weight prefill, seq 16 | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf_at_emitted_batch --tb=short -x` | 0.9999980480789986 |
| Synthetic decode, cache position 7 | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_decode_matches_hf_one_step_at_emitted_batch --tb=short -x` | 0.9990275510917438 |
| Real-weight decode, cache position 7 | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_decode_matches_hf_at_emitted_batch --tb=short -x` | 0.9999980525998797 |
| Full test file | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short` | 7 passed |

## Limitations

- The forge emit is a static batch-32 decode workload with cache length 128. This stage translates that decode path for correctness; it does not begin optimized decoder, multichip, full-model, or vLLM work.
- Prefill was validated at sequence lengths 16, 17, and 64. `doc/context_contract.json` records `current_supported_context=64`; the HF advertised 131072-token context is not claimed by this single-device functional stage because the emitted batch-32 BF16 hidden state alone would require 32 GiB before QKV, MLP intermediates, KV/cache, and op workspaces.
- Decode uses the emitted `paged_update_cache` and SDPA-decode path. The update and concat decode ops require height-sharded tensors; this implementation uses a local 8x4 core grid on the 1x1 device rather than copying the emit's wider grid ranges.
- `tt-smi -ls --local` could not be used in this environment because the installed script cannot import `tt_smi`; TTNN mesh open/close was exercised by the tests.
