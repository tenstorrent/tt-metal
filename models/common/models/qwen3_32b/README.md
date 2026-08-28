# Qwen3-32B with TTTv2

This directory contains the model-owned Qwen3-32B TTTv2 product path.
Qwen3-32B is currently the only Qwen3 model in `models/common/models`, so its
sampling and trace policy remains here rather than in a speculative Qwen3
family module.

## Product path

```text
Qwen/Qwen3-32B
  -> hf_adaptor.py: provider metadata, tokenizer, and weight conversion
  -> model.py: Qwen3 tensor graph composed from TTTv2 modules
  -> executor.py: Qwen3 policy over the shared model-layer ModelExecutor
  -> generator.py: vLLM construction, DP composition, and dispatch
```

## Files

| File | Responsibility |
| --- | --- |
| `hf_adaptor.py` | Resolve HF configuration/tokenizer/weights and construct runtime metadata |
| `weight_utils.py` | Convert and map provider weights |
| `model.py` | Build and execute the Qwen3-32B tensor graph |
| `executor.py` | Supply Qwen3-native sampling, trace-prime, and compatibility policy |
| `generator.py` | Build lanes, compose DP, configure `VLLMAdapter`, and dispatch calls |
| `demo.py` | Direct model demonstration entry point |

## Tensor-module composition

The model graph uses reusable TTTv2 modules for embedding, rotary setup,
RMSNorm, attention, MLP, LM head, and optional `Sampling1D`, together with
common TT collective helpers. Qwen3-specific QK normalization, dimensions,
precision, and device tuning remain in `model.py` and its configuration.

## Executor construction

`Qwen3_32BExecutor` is a composition facade over
`models/common/models/executor.py::ModelExecutor`. It does not subclass the
shared executor and does not duplicate its resource lifecycle.

The shared owner composes:

```text
Qwen3_32B model
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

The model-owned facade supplies only Qwen3-32B policy:

- caller-owned `SamplingState1D` state when device sampling is enabled;
- prompt history, output history, and slot-remap forwarding;
- sequential prefill while stateful device sampling is active;
- T3K trace-capture prime sequence lengths; and
- legacy eager/traced wrappers, trace bookkeeping, and direct-run helpers.

There is intentionally no `models/common/models/qwen3_executor.py`. A Qwen3
family layer should be introduced only after another Qwen3 model demonstrates
the same policy.

## vLLM and data parallelism

`Qwen3_32BGenerator` constructs one model/executor per lane. DP1 uses the
executor facade directly. DP greater than one uses `LaneGroupExecutor`, which
slices request state per lane and restores global output/logprob order.

`VLLMAdapter` normalizes server calls and validates the vLLM-selected paged-KV
shape. The executor lanes own TT resources; the generator and adapter own no TT
tensors.

## Ownership and cleanup

The composed `ModelExecutor` owns KV tensors, compile/trace registries, output
leases, sampling state, and deterministic cleanup. Cleanup terminalizes the
lane, drains external reads, releases prefill/decode transients and traces,
then releases sampling and KV resources with retryable failure reporting.

## Tests

Relevant entry points include:

- `models/common/tests/models/qwen3_32b/test_hf_adaptor.py`
- `models/common/tests/models/qwen3_32b/test_model_runtime_surface.py`
- `models/common/tests/models/qwen3_32b/test_module_profiles.py`
- `models/common/tests/models/qwen3_32b/test_demo_contract.py`
- `models/common/tests/models/qwen3_32b/test_p150x4_smoke.py`
- `models/common/tests/demos/qwen3_32b/demo.py`
- `models/common/tests/llm_runtime/test_executor_integration.py`
- `models/common/tests/llm_runtime/test_model_executor.py`
