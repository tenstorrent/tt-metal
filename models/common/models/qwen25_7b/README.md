# Qwen2.5-7B with TTTv2

This directory contains the Qwen2.5-7B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: Qwen2.5 tensor graph composed from TTTv2 modules
  -> executor.py: thin typed entry point into qwen2_executor.py
  -> generator.py: vLLM boundary, DP composition, and dispatch
```

`model.py` owns Qwen2.5 architecture/tuning while composing reusable embedding,
rotary, RMSNorm, attention, MLP, LM-head, optional sampling, and collective
modules.

## Executor composition

The model-local file preserves the Qwen2.5-7B executor/config/builder imports.
The implementation is shared with Qwen2 through
`models/common/models/qwen2_executor.py`, which configures the family-neutral
`ModelExecutor`.

The common owner composes paged KV, output reading, prefill/decode runtimes,
eager/trace execution, warmup, and cleanup. The family layer retains the
7B Q128 top-k warmup order and narrow pre-history request contract. It does not
silently adopt Qwen3 stateful sampling behavior.

## vLLM, DP, and ownership

`generator.py` builds one executor per lane, uses `VLLMAdapter` for external
normalization/cache validation, and composes multiple lanes with
`LaneGroupExecutor`. TT resources remain lane-owned.

## Tests

- `models/common/tests/models/qwen25_7b/test_hf_adaptor.py`
- `models/common/tests/models/qwen25_7b/test_demo_contract.py`
- `models/common/tests/demos/qwen25_7b/demo.py`
- `models/common/tests/models/test_qwen2_executor_family.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
