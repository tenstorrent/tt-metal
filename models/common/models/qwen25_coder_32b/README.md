# Qwen2.5-Coder-32B with TTTv2

This directory contains the Qwen2.5-Coder-32B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weights
  -> model.py: Qwen2.5-Coder tensor graph
  -> executor.py: thin family builder plus compatibility/direct-run helpers
  -> generator.py: vLLM construction, DP composition, and dispatch
```

The model uses reusable embedding, rotary, RMSNorm, attention, MLP, LM-head,
optional sampling, and collective modules. Coder architecture, dimensions,
precision, and tuning remain model-owned.

## Executor composition

The lifecycle is shared through `models/common/models/qwen2_executor.py`, which
configures `models/common/models/executor.py::ModelExecutor`. The common owner
constructs paged KV, output reading, prefill/decode runtimes, eager/traced
execution, warmup, and ordered cleanup.

The model-local file retains its historical executor/config/builder names,
eager/traced compatibility wrappers, compatibility config construction,
direct-run helpers, concrete last-token slicing, and TT deallocation behavior.
It uses plain coordinator prefill warmup and does not adopt Qwen3 native
sampling state during this structural extraction.

## vLLM, DP, and ownership

The generator owns vLLM normalization/dispatch and constructs one executor per
lane. `LaneGroupExecutor` provides DP fanout. Lane executors exclusively own TT
resources and cleanup.

## Tests

- `models/common/tests/models/qwen25_coder_32b/test_hf_adaptor.py`
- `models/common/tests/models/qwen25_coder_32b/test_model_runtime_surface.py`
- `models/common/tests/models/qwen25_coder_32b/test_demo_contract.py`
- `models/common/tests/demos/qwen25_coder_32b/demo.py`
- `models/common/tests/models/test_qwen2_executor_family.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
