# Llama 3.2 1B with TTTv2

This directory contains the Llama 3.2 1B TTTv2 product path.

## Construction path

```text
HF checkpoint
  -> hf_adaptor.py: provider configuration, tokenizer, and weights
  -> model.py: TTTv2 Llama tensor graph
  -> executor.py: thin typed entry point into llama3_executor.py
  -> generator.py: vLLM boundary, lane construction, and dispatch
```

`model.py` composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, and optional sampling modules. Llama 3.2 1B architecture and tuning
remain model-owned.

## Executor composition

The model-local `executor.py` exports the historical
`Llama32_1BExecutor`/config/builder names. The implementation lives in
`models/common/models/llama3_executor.py`, which supplies Llama-family Q128
warmup policy and composes `models/common/models/executor.py::ModelExecutor`.

The shared executor owns:

- `PagedKVCacheManager` and `PageTableLayout`;
- `OutputReader`;
- `PrefillRuntime` and `DecodeRuntime`;
- `ProgramCompiler` and `EagerExecutor`;
- optional `TraceCompiler`/`TracedExecutor`;
- `WarmupCoordinator`; and
- terminal, ordered cleanup.

Llama 3.2 1B preserves its narrow pre-history request signatures, current
sampling behavior, and Q128 priming-before-prefill order. It does not silently
adopt the newer Llama-8B/70B `SamplingState1D` lifecycle in this structural
refactor.

## vLLM, DP, and ownership

`generator.py` builds one model/executor per lane and configures
`VLLMAdapter`. DP1 uses the executor directly; larger DP uses
`LaneGroupExecutor`. Executors own TT resources; the generator/adapter own
dispatch policy only.

## Tests

- `models/common/tests/models/llama32_1b/test_hf_adaptor.py`
- `models/common/tests/models/llama32_1b/test_batched_prefill_postprocess.py`
- `models/common/tests/models/llama32_1b/test_demo_warmup.py`
- `models/common/tests/demos/llama32_1b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
