# Qwen2-7B with TTTv2

This directory contains the Qwen2-7B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: Qwen2 tensor graph composed from TTTv2 modules
  -> executor.py: thin typed entry point into qwen2_executor.py
  -> generator.py: vLLM boundary, DP composition, and dispatch
```

The model composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, optional sampling, and TT collective modules. Qwen2 architecture and
device tuning remain model-owned.

## Executor composition

The local `executor.py` retains the public Qwen2 class/config/builder names.
`models/common/models/qwen2_executor.py` supplies Qwen2-family policy and
composes `models/common/models/executor.py::ModelExecutor`.

The shared owner constructs paged-KV management, output reading,
prefill/decode runtimes, eager/trace execution, warmup, and ordered cleanup.
The family layer preserves Qwen2-7B's narrow request signatures and Q128
top-k tile-end priming before traced warmup and after eager warmup. This
structural refactor does not add Qwen3 native sampling state.

## vLLM, DP, and ownership

The generator builds one executor per lane and configures `VLLMAdapter`. DP1
uses one lane; larger DP uses `LaneGroupExecutor`. Executors own TT resources;
the generator and adapter own dispatch policy only.

## Tests

- `models/common/tests/models/qwen2_7b/test_hf_adaptor.py`
- `models/common/tests/models/qwen2_7b/test_demo_contract.py`
- `models/common/tests/demos/qwen2_7b/demo.py`
- `models/common/tests/models/test_qwen2_executor_family.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
