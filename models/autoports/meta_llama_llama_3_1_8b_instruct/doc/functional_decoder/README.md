# Llama 3.1 8B Instruct Forge Functional Decoder

This stage translates the decoder-layer math from `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0` into `models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py`.

## Provenance

Translated files:

- `model_ttnn.py`: layer, attention, MLP, RMSNorm, RoPE, projection, and residual semantics.
- `consteval.py`: setup-time projection transposes, fused QKV construction intent, and RoPE/cache constants.
- `params.py`: canonical HF weight-key mapping.
- `model_pt.py`: emitted workload model id, batch size `32`, and source harness context.

The implementation intentionally does not copy static shard specs, paged-cache update code, per-core grids, or matmul program configs from the emit.

## Runtime Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len=128, batch=32, **kwargs)` accepts canonical HF keys such as `model.layers.{i}.self_attn.q_proj.weight`, plus layer-local and `model.language_model.layers.{i}` forms. It performs setup-time conversion to TTNN BF16/TILE/DRAM tensors, fuses QKV in `[V, Q, K]` order, and builds RoPE tables plus a causal mask for `max_seq_len`.

`prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None)` expects a TTNN tensor shaped `[1, batch, seq_len, 4096]` and returns `[1, batch, seq_len, 4096]`. Runtime prefill is TTNN-only: no torch conversion, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

`decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Validation

| Check | Command | Result |
| --- | --- | --- |
| Static runtime fallback audit | `pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k runtime_has_no_host_fallback --tb=short` | Passed |
| Synthetic prefill, batch 32, seq 16 and 64 plus decode stub | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k 'synthetic_weight_prefill or decode_forward_documents_pending_path' --tb=short` | Passed: seq 16 PCC `0.9985012578882686`; seq 64 PCC `0.9981899661680298` |
| Real-weight single-layer prefill, batch 32, seq 16 | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k real_weight_single_layer_prefill_matches_hf --tb=short` | Passed: PCC `0.9999980524146453` |
| Full functional decoder tests | `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short` | Passed: 5 tests in 30.42s; real-weight PCC `0.9999980386971122` |
| Context contract | `.agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct` | Passed; checker output labels reduced context as `DRAM-limited` because of its current schema |

## Limitations

- Prefill only.
- The source forge graph in this checkout is decode/cache-oriented after an HF prefill, while this stage produces only the requested generalized single-layer prefill path.
- Decode pending a future emitted-decode forge version.
- No paged KV cache.
- Current supported context is below the HF-advertised 131072 positions; this is the largest validated prefill length in this stage, not a proven physical limit.
- `context_contract.json` includes a checker-compatibility `capacity_evidence` block because the local checker requires that field for reduced context; it does not claim a measured DRAM limit.
- Warmed prefill timing has not been measured in this pass.
