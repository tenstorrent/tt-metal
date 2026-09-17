# Qwen2.5-72B with TTTv2

This directory contains the large Qwen2.5-72B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weights
  -> model.py: multi-device Qwen2.5 tensor graph
  -> executor.py: thin family builder plus concrete direct-run helpers
  -> generator.py: vLLM construction, DP composition, and dispatch
```

The tensor model composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, optional sampling, and collective modules while retaining 72B
topology, precision, and program policy locally.

## Executor composition

The public 72B executor/config/builder remains importable from this directory.
Its lifecycle is supplied by `models/common/models/qwen2_executor.py` over
`models/common/models/executor.py::ModelExecutor`.

The shared owner composes paged-KV management, prefill/decode runtimes,
eager/trace compilation, output handling, warmup, and cleanup. Qwen2.5-72B uses
plain coordinator prefill warmup rather than the 7B Q128 priming hook.

Concrete `run_prefill`, `run_decode`, `run_lm_head`, last-token slicing, and
deallocation helpers remain model-local because they invoke the concrete
tensor model.

## vLLM, DP, and ownership

The generator builds one executor per lane, configures `VLLMAdapter`, and uses
`LaneGroupExecutor` for DP. Each lane owns its KV tensors, compile/trace
artifacts, output leases, sampling buffers, and terminal cleanup.

## Tests

- `models/common/tests/models/qwen25_72b/test_hf_adaptor.py`
- `models/common/tests/models/qwen25_72b/test_model_runtime_surface.py`
- `models/common/tests/models/qwen25_72b/test_demo_contract.py`
- `models/common/tests/demos/qwen25_72b/demo.py`
- `models/common/tests/models/test_qwen2_executor_family.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
