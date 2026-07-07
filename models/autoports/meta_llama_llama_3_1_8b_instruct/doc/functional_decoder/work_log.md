# Llama-3.1-8B-Instruct Functional Decoder Work Log

## Stage

- Skill: `forge-functional-decoder`
- Hardware practice: `tt-device-usage` for serialized TTNN/device-facing commands.
- Debugging loop: `autofix` run after the real-weight PCC gate skipped on gated HF access.
- Target: `meta-llama/Llama-3.1-8B-Instruct`
- Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`
- Forge source: `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`

## Translation Notes

- Read `model_ttnn.py`, `consteval.py`, `params.py`, `model_pt.py`, and `main.py`.
- Implemented a single-layer `FunctionalDecoder(LightweightModule)`.
- Preserved Llama decoder math: RMSNorm eps `1e-5`, fused projection semantics, RoPE via `ttnn.experimental.rotary_embedding`, SDPA scale `1/sqrt(128)`, raw HF projection weights with `transpose_b=True` for non-fused projections, and SwiGLU MLP.
- Llama-specific differences from the Qwen forge notes: no q/k per-head RMSNorm; hidden size `4096`; intermediate size `14336`; 32 Q heads and 8 KV heads; `max_position_embeddings=131072`; Llama3 RoPE scaling with `rope_theta=500000`.
- The Llama emit is a one-token decode/cache graph. It uses batch-32 static cache glue, `paged_update_cache`, `scaled_dot_product_attention_decode`, and `nlp_concat_heads_decode`; those are not copied into this prefill-only functional decoder.
- `decode_forward` is an intentional `NotImplementedError` stub pending an emitted decode graph for this stage.

## Commands And Evidence

Hardware-facing commands were run serially per `tt-device-usage`.

```bash
timeout 60 tt-smi -ls --local
```

Result: failed before device use because `tt-smi` is not installed or on PATH in this environment.

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK 1x1')
PY
```

Result: passed; printed `MESH_SMOKE_OK 1x1`.

Host/static checks:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
```

Result: passed.

```bash
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -k runtime_has_no_host_fallback --tb=short
```

Result: passed in 1.72s.

Prefill checks:

```bash
pytest -q -s 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[16]' --tb=short
```

Result: passed in 20.18s. Synthetic seq 16 PCC `0.9984283954871392`.

```bash
pytest -q -s 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[17]' 'models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer[64]' --tb=short
```

Result: passed in 23.63s. Synthetic seq 17 PCC `0.9984148482493451`; synthetic seq 64 PCC `0.9982231899002421`.

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_decode_forward_documents_pending_path --tb=short
```

Result: passed in 7.32s.

Real-weight gate:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

Result before test-harness update: skipped in 2.32s because `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")` returned a gated-repo `401`. This was not accepted as a completed gate.

After AutoFix added the explicit local-checkout override and skip-before-mesh behavior:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

Result: skipped in 2.15s with message to set `LLAMA31_8B_INSTRUCT_HF_PATH` to a canonical local HF checkout or provide HF auth. No TT mesh was opened before the skip.

Resumed run after HF authentication became available:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

Result: passed in 93.20s. Real-weight single-layer prefill PCC `0.9999980443319924`.

Final full-file snapshot after real-weight access was available:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short
```

Result: `6 passed in 30.18s`. PCCs in this run: synthetic seq 16 `0.9984283954871392`, synthetic seq 17 `0.9984124017817244`, synthetic seq 64 `0.9982231899002421`, real-weight seq 16 `0.9999981161772881`.

Exploratory local cache probe:

```bash
python - <<'PY'
# loaded several /localdev/hmijatovic/model_cache/.../*.tensorbin files with ttnn.load_tensor
PY
```

Result: `ttnn.load_tensor` unexpectedly opened UMD and closed it cleanly. The candidate files were distributed/quantized TT transformer cache shards and not directly convertible to canonical HF weights without a mesh composer and layout-specific reconstruction. They were not used as stage evidence.

## PCC

- Synthetic seq 16: `0.9984283954871392`.
- Synthetic seq 17: `0.9984148482493451` in targeted run; `0.9984124017817244` in final full-file run.
- Synthetic seq 64: `0.9982231899002421`.
- Real weights: `0.9999980443319924` in the targeted required command; `0.9999981161772881` in the final full-file run.

## Runtime Contract

- `from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len=128, **kwargs)`
- `prefill_forward(hidden_states, *, position_cos=None, position_sin=None, attention_mask=None) -> ttnn.Tensor`
- Input shape: `[1, 1, seq_len, 4096]`
- Output shape: `[1, 1, seq_len, 4096]`
- Runtime `prefill_forward` contains no torch conversion, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.
- `decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version")`.

## Limitations

- Prefill-only.
- Provided forge origin is static one-token decode/cache; this pass generalizes setup-time RoPE and mask construction for prefill but validates only selected lengths.
- Decode pending a future emitted-decode forge version.
- No paged KV path.
- Current supported context is below the HF-advertised 131072 context.
- Real-weight PCC evidence depends on HF authentication or a canonical local HF checkout. The test supports `LLAMA31_8B_INSTRUCT_HF_PATH` for unauthenticated environments.

## AutoFix

- Fresh AutoDebug report: `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/AUTODEBUG.md`.
- AutoFix report: `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/AUTOFIX.md`.
- Verdict before resume: blocked by external credential/data availability.
- Resumed status: HF authentication became available and the required real-weight single-layer PCC gate passed.
- Refuted local alternatives: default HF cache, forge-generated tree, repo-local config mirrors, TT transformer `.tensorbin` caches, and canonical Llama shard names under `/localdev`.
- Portability note: provide HF credentials for `meta-llama/Llama-3.1-8B-Instruct` or set `LLAMA31_8B_INSTRUCT_HF_PATH` to a canonical local HF checkpoint checkout.

## Stage Review

- `$stage-review` subagent `019f3b70-1504-7870-9ee8-8ca459d32775` returned `clean-pass`.
