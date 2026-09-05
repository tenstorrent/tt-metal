# DeepSeek-R1-Distill-Qwen-14B with TTTv2

This directory is a model-owned TTTv2 product path for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.

It intentionally demonstrates direct executor construction from the reusable
common LLM runtime. It is not part of the shared Llama/Qwen executor
consolidation and does not use `models/common/models/executor.py`.

## Product path

```text
Hugging Face checkpoint
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: DeepSeek-Qwen tensor graph composed from TTTv2 modules
  -> executor.py: direct composition of common runtime owners for one lane
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve the HF checkpoint/tokenizer and construct model/runtime configuration |
| `weight_utils.py` | Convert and map provider weights into the model-owned layout |
| `model.py` | Build and execute the DeepSeek-Qwen transformer graph |
| `executor.py` | Directly compose one lane from `llm_runtime` modules and own cleanup |
| `generator.py` | Build one or more lanes, configure `VLLMAdapter`, and expose the vLLM-facing API |

## Tensor-module composition

`model.py` composes the model from reusable TTTv2 modules, including:

- `Embedding1D`
- `RotarySetup1D`
- `RMSNorm1D`
- `Attention1D`
- `MLP1D`
- `LMHead1D`
- optional `Sampling1D`
- common TT collective helpers

The model owns DeepSeek/Qwen architecture and tuning policy. The reusable
modules own their tensor programs and lazy weights.

## Direct executor composition

`DeepSeekR1Qwen14BExecutor` directly constructs:

```text
DeepSeekR1Qwen14B model
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

This direct pattern is supported when a model has genuinely distinct
orchestration or has not been deliberately migrated to a shared model-layer
executor. It still reuses the same `models/common/llm_runtime` mechanics; it
does not copy those runtime implementations.

The executor owns paged-KV tensors, eager/trace registries, output leases,
sampling buffers, and deterministic cleanup. Cleanup terminalizes the lane,
drains external decode outputs, releases runtime transients and traces, then
releases sampling and KV resources with retryable failure reporting.

## vLLM and data parallelism

`DeepSeekR1Qwen14BGenerator` builds one executor per lane. DP1 uses the lane
directly; DP greater than one wraps lanes in `LaneGroupExecutor`.

`VLLMAdapter` normalizes vLLM calls and validates the physical KV-cache shape.
The generator owns dispatch policy but no TT tensor resources.

## Tests

Relevant entry points include:

- `models/common/tests/models/deepseek_r1_distill_qwen_14b/test_hf_adaptor.py`
- `models/common/tests/models/deepseek_r1_distill_qwen_14b/test_demo_contract.py`
- `models/common/tests/models/deepseek_r1_distill_qwen_14b/test_prefill_last_token_contract.py`
- `models/common/tests/demos/deepseek_r1_distill_qwen_14b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
