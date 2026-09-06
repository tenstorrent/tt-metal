# Mistral-7B with TTTv2

This directory is the model-owned TTTv2 path for the Mistral-7B family.

It intentionally demonstrates direct executor construction from
`models/common/llm_runtime`. It is not part of the Llama/Qwen executor
consolidation and does not use `models/common/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Mistral tensor graph composed from TTTv2 modules
  -> executor.py: direct composition of common runtime owners for one lane
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve provider configuration/tokenizer and construct the product model |
| `weight_utils.py` | Convert and map provider weights |
| `model.py` | Build and execute the TTTv2 Mistral transformer graph |
| `executor.py` | Directly compose one execution lane and own its resources |
| `generator.py` | Build lanes, configure the vLLM boundary, and select eager/traced execution |

## Tensor-module composition

`model.py` composes:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D`
- `MLP1D`
- `LMHead1D`
- optional `Sampling1D`
- common TT collective helpers

Mistral-specific attention, RoPE, precision, and device-tuning policy remains
model-owned.

## Direct executor composition

`Mistral7BExecutor` directly constructs:

```text
Mistral7B model
├── PagedKVCacheManager
├── OutputReader
├── PrefillRuntime
├── DecodeRuntime
├── ProgramCompiler
├── EagerExecutor
├── optional TraceCompiler
├── optional TracedExecutor
└── WarmupCoordinator
```

This is a supported alternative to the shared model-layer `ModelExecutor`.
Models with distinct orchestration may compose the focused `llm_runtime`
modules directly without subclassing or modifying a universal executor.

The lane executor owns paged KV, compile/trace registries, output leases,
sampling buffers, and deterministic cleanup. The generator owns orchestration
only and does not own TT tensors.

## vLLM and data parallelism

`Mistral7BGenerator` builds one model/executor per lane and uses
`LaneGroupExecutor` when `tt_data_parallel > 1`. `VLLMAdapter` normalizes the
server boundary and validates the vLLM-selected KV-cache specification.

## Tests

Relevant entry points include:

- `models/common/tests/models/mistral_7b/test_hf_adaptor.py`
- `models/common/tests/models/mistral_7b/test_demo_contract.py`
- `models/common/tests/models/mistral_7b/test_prefill_last_token_contract.py`
- `models/common/tests/demos/mistral_7b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
