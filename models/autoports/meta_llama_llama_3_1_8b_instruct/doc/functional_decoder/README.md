# Llama-3.1-8B-Instruct Forge Functional Decoder

This stage translates the decoder-layer math from `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0` into `models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py`.

## Provenance

Translated files:

- `model_ttnn.py`: Llama decoder layer residual structure, RMSNorm, attention projection, RoPE, output projection, and SwiGLU MLP semantics.
- `consteval.py`: setup-time projection transposes, fused QKV layout, `model.rotary_emb.inv_freq` handling, and static decode/cache constants.
- `params.py`: traced `.parametrizations.weight.original` names mapped back to canonical HF `.weight` keys.
- `model_pt.py` and `main.py`: source model id, bf16 dtype, batch/cache shape, and evidence that the emitted graph is one-token decode after prefill.

The implementation intentionally does not copy batch-32/cache-128 reshape glue, paged-cache updates, `scaled_dot_product_attention_decode`, tuned shard specs, or matmul program configs from the emit.

## Runtime Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len=128, **kwargs)` accepts canonical HF keys such as `model.layers.{i}.self_attn.q_proj.weight`, plus layer-local and `model.language_model.layers.{i}` forms. It performs setup-time conversion to TTNN BF16/TILE/DRAM tensors, fuses QKV in the Llama emit order, and builds RoPE tables plus a causal mask for `max_seq_len`.

`prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None)` expects a TTNN tensor shaped `[1, 1, seq_len, 4096]` and returns `[1, 1, seq_len, 4096]`. Runtime prefill is TTNN-only: no torch conversion, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

`decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Validation

| Check | Command | Result |
| --- | --- | --- |
| Syntax | `python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py` | Passed |
| Static runtime fallback audit | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k runtime_has_no_host_fallback --tb=short` | Passed |
| Mesh smoke | `python - <<'PY' ... open_mesh_device(MeshShape(1, 1)) ... PY` | Passed; printed `MESH_SMOKE_OK 1x1` |
| Synthetic prefill, seq 16 | `pytest -q -s 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[16]' --tb=short` | Passed; PCC `0.9984283954871392` |
| Synthetic prefill, seq 17 and 64 | `pytest -q -s 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[17]' 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[64]' --tb=short` | Passed; PCCs `0.9984148482493451`, `0.9982231899002421` |
| Decode stub | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_decode_forward_documents_pending_path --tb=short` | Passed |
| Real-weight single-layer prefill | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short` | Passed; PCC `0.9999980443319924` |
| Full test file | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short` | `6 passed in 30.18s`; synthetic PCCs `0.9984283954871392`, `0.9984124017817244`, `0.9982231899002421`; real-weight PCC `0.9999981161772881` |

## Limitations

- Prefill only.
- The provided forge emit is a static one-token decode/cache graph, not the static prefill graph described in the stage prompt. This artifact translates the shared decoder-layer math into prefill and leaves decode pending.
- `decode_forward` is a documented pending stub until an emitted decode version is supplied for this autoport stage.
- No paged KV path.
- Current validated context is 64, below the HF-advertised 131072 positions. This is a stage validation limit, not a proven physical DRAM limit.
- Real-weight validation used HF authentication-backed access to `meta-llama/Llama-3.1-8B-Instruct`. In environments without that access, set `LLAMA31_8B_INSTRUCT_HF_PATH` to a canonical local HF checkout.
- Warmed prefill timing has not been measured in this pass.
