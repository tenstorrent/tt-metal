# Llama 3.3 70B with TTTv2

This directory contains the Llama 3.3 70B TTTv2 product path.

## Construction path

```text
meta-llama/Llama-3.3-70B-Instruct
  -> hf_adaptor.py: provider metadata, tokenizer, and weights
  -> model.py: multi-device TTTv2 Llama graph
  -> executor.py: thin typed entry point into llama3_executor.py
  -> generator.py: vLLM construction, DP lanes, and dispatch
```

`model.py` composes reusable embedding, rotary, RMSNorm, attention, MLP,
LM-head, sampling, and TT collective modules. The 70B topology, precision, and
program policy remains model-owned.

## Executor composition

The local file retains `Llama33_70BExecutor`, its config, and its builder as
stable imports. `models/common/models/llama3_executor.py` supplies the 70B
family policy and composes `models/common/models/executor.py::ModelExecutor`.

The family-neutral owner composes paged-KV management, output reading,
prefill/decode runtimes, eager and traced execution, warmup coordination, and
ordered cleanup. The Llama family layer adds only:

- `SamplingState1D` and complete request history/remap state;
- device-sampling prefill serialization;
- the 70B trace-warmup sequence preference; and
- Q128 top-k priming before traced warmup and after eager warmup.

## vLLM, DP, and ownership

The generator builds one model/executor per submesh. DP1 uses one executor;
larger DP uses `LaneGroupExecutor`. `VLLMAdapter` normalizes server calls and
validates the physical KV cache. Each lane owns its TT resources and sampling
state; generator/adapter objects own no TT tensors.

## Tests

- `models/common/tests/models/llama33_70b/test_hf_adaptor.py`
- `models/common/tests/models/llama33_70b/test_model_profile.py`
- `models/common/tests/models/llama33_70b/test_demo_contract.py`
- `models/common/tests/models/llama33_70b/test_logits_oracle.py`
- `models/common/tests/models/llama33_70b/test_t3k_batched_prefill_correctness.py`
- `models/common/tests/models/llama33_70b/test_p150x4_smoke.py`
- `models/common/tests/demos/llama33_70b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
