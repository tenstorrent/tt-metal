# Functional Decoder Work Log

## 2026-07-13

Target: `meta-llama/Llama-3.1-8B-Instruct`

Forge emit: `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct`

Branch at start: `mvasiljevic/shard-advise-fresh`

Start commit: `b9f246842170644d74d82c39f0b12b39cb8853c0`

## Emit Inspection

- `model_pt.py` records `BATCH_SIZE = 32`, `INPUT_SEQUENCE_LENGTH = 128`, and a StaticCache decode workload.
- Grep found decode ops in the emit: `paged_update_cache`, `scaled_dot_product_attention_decode`, and `nlp_concat_heads_decode`.
- A sidecar read-only audit found the actual emit QKV fusion order is `[Q,V,K]` for layers 0-30 and `[Q,K,V]` for layer 31. This contradicts the prompt summary's `[V,Q,K]`; the implementation follows the actual emitted files.
- The emit and params inventory contain no per-head Q/K norm weights. The implementation has optional q/k norm support for compatible state dicts, but the Llama path does not use it.
- RMSNorm epsilon is `1e-5`, matching HF Llama config and `model_ttnn.py`.

## Implementation Notes

- Created `models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py`.
- Added package files and tests under the same autoport tree.
- `from_state_dict` accepts canonical HF keys and layer-local keys, fuses emitted QKV weights, builds RoPE tables, and moves weights to TTNN. Default prefill uses SDPA causal mode without a materialized causal mask.
- `prefill_forward` runs RMSNorm, emitted QKV split, RoPE, causal SDPA, output projection, residual, post-attention RMSNorm, SwiGLU MLP, and final residual.
- `decode_forward` translates the emitted single-token graph: RoPE-at-position, `paged_update_cache` for K/V, SDPA decode with mask and scale, `nlp_concat_heads_decode`, output projection, residual, post-attention RMSNorm, SwiGLU MLP, and final residual.
- Runtime forwards and runtime helpers have no `torch`, `ttnn.from_torch`, or `ttnn.to_torch`.

## TT Device Notes

- `timeout 60 tt-smi -ls --local` failed with `ModuleNotFoundError: No module named 'tt_smi'`.
- TTNN tests successfully opened and closed a `MeshShape(1, 1)` device. Logs showed local chip `{0}` and remote chip `{1}` visible through UMD.
- No reset was performed.

## Validation Commands

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_runtime_forwards_have_no_host_fallback
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_prefill_matches_hf_layer_at_emitted_batch --tb=short -x
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_synthetic_weight_decode_matches_hf_one_step_at_emitted_batch --tb=short -x
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf_at_emitted_batch --tb=short -x
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_decode_matches_hf_at_emitted_batch --tb=short -x
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short
```

Latest full test result after the context-contract fix:

```text
7 passed in 40.93s
Synthetic prefill PCCs: 0.9984961177648134, 0.9984850444029093, 0.9981846012611152
Real prefill PCC: 0.9999980480789986
Synthetic decode PCC: 0.9990275510917438
Real decode PCC: 0.9999980525998797
```

## Context Contract

- Initial review found `current_supported_context=131072` overclaimed full context because only seq 16/17/64 were validated and the setup path built an unused full causal mask.
- Fixed by removing the unused setup causal-mask allocation from `from_state_dict`.
- Updated `doc/context_contract.json` to record `current_supported_context=64`, with device-DRAM capacity evidence for why the emitted batch-32 BF16 single-device functional path does not claim the HF advertised 131072-token context.
- `.agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct` passes with `target=131072, supported=64 (DRAM-limited)`.

## Pending Gates

- Independent `$stage-review` rereview returned `clean-pass` with no required work.
- Commit stage-owned changes locally.

## Stage Review

Initial review returned `more-work-needed` for the context contract overclaim. After remediation, fresh rereview returned:

```text
Verdict: clean-pass
Required Work: None
Residual risk: decode cache update would benefit from a future prefix-cache-only assertion, but the current implementation and stage contract are sufficient for clean pass.
```

## Local Commit

Repo: `/localdev/mvasiljevic/tt-metal`

Branch: `mvasiljevic/shard-advise-fresh`

Stage checkpoint commit: `0803a93063948ce6e375e5af6cfe9a7ddc279a39`
