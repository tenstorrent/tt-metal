# Llama 3.2 3B with TTTv2

This directory contains the Llama 3.2 3B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: TTTv2 Llama tensor graph
  -> executor.py: thin typed entry point into llama3_executor.py
  -> generator.py: vLLM boundary, lane construction, and dispatch
```

The tensor graph composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, and optional sampling modules. The 3B architecture and tuning remain
in this model package.

## Executor composition

The model-local `executor.py` retains the public
`Llama32_3BExecutor`/config/builder names while delegating to
`models/common/models/llama3_executor.py`. The family module configures the
family-neutral `models/common/models/executor.py::ModelExecutor`.

That owner composes paged-KV management, output reading, prefill/decode
runtimes, eager/trace compilation, warmup, and deterministic cleanup from
`models/common/llm_runtime`.

Llama 3.2 3B preserves the same narrow request signatures and Q128
priming-before-prefill policy as before the extraction. It does not gain the
newer Llama-8B/70B native sampling-state contract as a side effect.

## vLLM, DP, and ownership

The generator builds one executor per lane, uses `VLLMAdapter` at the server
boundary, and wraps lanes with `LaneGroupExecutor` for DP. Lane executors own
all TT tensors and cleanup; the generator owns no TT resources.

## Tests

- `models/common/tests/models/llama32_3b/test_hf_adaptor.py`
- `models/common/tests/models/llama32_3b/test_batched_prefill_postprocess.py`
- `models/common/tests/models/llama32_3b/test_demo_warmup.py`
- `models/common/tests/demos/llama32_3b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
