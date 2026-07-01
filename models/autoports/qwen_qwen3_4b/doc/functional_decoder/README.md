# Qwen3-4B Forge Functional Decoder

This stage translates the prefill math from `/home/ubuntu/ttnn-models/Qwen/Qwen3-4B/model/graph_0` into `models/autoports/qwen_qwen3_4b/tt/functional_decoder.py`.

## Provenance

Translated files:

- `model_ttnn.py`: layer, attention, MLP, RMSNorm, RoPE, and SDPA semantics.
- `consteval.py`: fused QKV order `[V, Q, K]`, RoPE table generation, and causal-mask construction.
- `params.py`: canonical HF weight-key mapping and forge weight layout intent.

The implementation intentionally does not copy seq_len=16 shard specs or matmul program configs from the emit.

## Runtime Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len=128, **kwargs)` accepts canonical HF keys such as `model.layers.{i}.self_attn.q_proj.weight`, plus layer-local and `model.language_model.layers.{i}` forms. It performs setup-time conversion to TTNN BF16/TILE/DRAM tensors, fuses QKV in `[V, Q, K]` order, and builds RoPE tables plus a causal mask for `max_seq_len`.

`prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None)` expects a TTNN tensor shaped `[1, 1, seq_len, 2560]` and returns `[1, 1, seq_len, 2560]`. Runtime prefill is TTNN-only: no torch conversion, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

`decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Validation

| Check | Command | Result |
| --- | --- | --- |
| Static runtime fallback audit | `pytest -q models/autoports/qwen_qwen3_4b/tests/test_functional_decoder.py -k runtime_has_no_host_fallback` | Passed |
| Synthetic prefill, seq 16 and 64 plus decode stub | `pytest -q models/autoports/qwen_qwen3_4b/tests/test_functional_decoder.py -k 'synthetic_weight_prefill or decode_forward_documents_pending_path'` | Passed: seq 16 PCC `0.9996239896537468`; seq 64 PCC `0.9996795472223063` |
| Real-weight single-layer prefill, seq 16 | `pytest -q models/autoports/qwen_qwen3_4b/tests/test_functional_decoder.py -k real_weight_single_layer_prefill_matches_hf` | Passed: PCC `0.9999931920726277` |
| Full functional decoder test file | `pytest -q models/autoports/qwen_qwen3_4b/tests/test_functional_decoder.py` | Passed: 5 tests in 25.32s; real-weight PCC `0.9999917295183249` |
| Context contract | `.agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b` | Passed; checker output labels reduced context as `DRAM-limited` because of its current schema |

## Limitations

- Prefill only.
- The source forge graph is static-shape prefill with no emitted decode path.
- No paged KV cache.
- Current validated context is below the HF-advertised 40960 positions; this is a stage validation limit, not a proven physical limit.
- `context_contract.json` includes a checker-compatibility `capacity_evidence` block because the local checker requires that field for reduced context; it does not claim a measured DRAM limit.
- Warmed prefill timing has not been measured in this pass.
