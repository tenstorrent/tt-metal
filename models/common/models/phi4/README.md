# Phi-4 with TTTv2

This directory is the model-owned TTTv2 path for Microsoft Phi-4.

It intentionally demonstrates direct executor construction from
`models/common/llm_runtime`. It is not part of the Llama/Qwen executor
consolidation and does not use `models/common/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Phi-4 tensor graph composed from TTTv2 modules
  -> executor.py: direct composition of common runtime owners for one lane
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve the Phi-4 provider configuration/tokenizer and build runtime metadata |
| `weight_utils.py` | Convert and map HF weights into the model-owned layout |
| `model.py` | Build and execute the TTTv2 Phi-4 transformer graph |
| `executor.py` | Directly compose one execution lane and own its runtime resources |
| `generator.py` | Build lanes, configure `VLLMAdapter`, and expose serving dispatch |

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

Phi-specific provider mapping, architecture values, precision, and tuning
remain model-owned.

## Direct executor composition

`Phi4Executor` directly constructs:

```text
Phi4Transformer
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

This direct construction path is intentionally retained as an example for new
models whose orchestration does not belong in the shared Llama or Qwen family
composition. It reuses focused runtime modules without adding a protocol,
profile, or executor subclass hierarchy.

The executor owns the physical KV cache, compile/trace artifacts, output
leases, sampling resources, and cleanup. Cleanup is terminal, ordered,
retryable, and idempotent.

## vLLM and data parallelism

`Phi4Generator` builds one executor per lane. DP1 uses the executor directly;
larger DP configurations use `LaneGroupExecutor`. `VLLMAdapter` performs
server-boundary normalization and KV-cache validation.

## Tests

Relevant entry points include:

- `models/common/tests/models/phi4/test_hf_adaptor.py`
- `models/common/tests/models/phi4/test_demo_contract.py`
- `models/common/tests/demos/phi4/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
