# Full Model Runtime Fallback Audit

## Selected Runtime

The full model uses `QwenFullModel` plus the completed optimized
`MultichipDecoder` stack. The selected runtime remains the local `2x2` p300c
mesh with tensor parallelism over mesh columns, expert parallelism over mesh
rows, BF16 residual boundaries, BF16 KV/cache state, inherited optimized
decoder weight dtypes/fidelity, and the BF8 flat 4-way LM head.

## Model And Generator Boundaries

| Boundary | Status |
| --- | --- |
| Decoder stack | `MultichipDecoder` for every layer; no single-chip decoder fallback. |
| Embedding/final norm/LM head | TTNN device path. Embedding and final norm are replicated BF16; LM head is flat vocab-sharded over four devices. |
| Cache ownership | `QwenFullModelCache` owns full-attention KV, linear state, page table, host page-table mirror, max batch, max sequence, and block size. |
| Page table | Public caller/generator supplies or receives explicit page-table state. Prefill selects the active row and chunk rows for fixed-slot cache fill. Decode reuses the persistent table. |
| Mixed prompts | Active rows update only their fixed cache slot. Inactive rows return zero logits and do not run decoder layers. |
| Host logits | Full-logit readback is used by readiness checks and explicit host-sampling compatibility only. It is not used by the optimized token-out measured path. |
| Sampling | Optimized greedy token-out uses common on-device `SamplingGenerator`, writes into a tile-width TT sampler output buffer, and copies output slot 0 into a separate 1-wide TT decode input buffer inside the trace. |
| Reset | `QwenReadinessGenerator.reset()` releases trace, clears cache ownership, and resets sampler state; `teardown()` calls `reset()`. |

The source audit strings exposed by `iter_runtime_fallback_audit()` cover the
selected model/generator path:

- `decoder_stack=MultichipDecoder`
- `mesh=(2,2); tensor_parallel_axis=columns; expert_parallel_axis=rows`
- `residual_layout=replicated_bf16_inter_layer`
- `embedding=replicated_bf16`
- `lm_head=vocab_sharded_flat_4way`
- `kv_cache=paged_bf16_split_full_attention`
- `linear_state=bf16_split_linear_attention`
- `sampling=common_sampling_generator_flat_4way_topk1_composite_gather`
- `no_single_chip_or_host_decoder_fallback`

## Dynamic Fallback Gate

```bash
timeout 1800 env TT_METAL_WATCHER_DISABLE_ETH=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  RUN_QWEN36_FULL_MODEL_SMOKE=1 \
  ./python_env/bin/python -m pytest -q \
  --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b/tests \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/logs/runtime_fallback_audit_synthetic.log
```

Result: `6 passed, 2 warnings in 46.56s`.

This gate covers the model wrapper, generator contract, non-aligned prefill,
mixed fixed slots with inactive rows, changed page-table decode override,
traced token-out, and host compatibility comparison with fallback exceptions
enabled.

## Host-Sampling Compatibility

`QwenReadinessGenerator.host_sampling_compat` intentionally exposes a
compatibility mode for tests that require host sampling, including
teacher-forcing readiness callbacks. This mode is not used for optimized
token-out measurement. In compatibility mode, host argmax is explicit and the
path is excluded from the optimized decode t/s/u claim.

## Rejected Or Non-Signoff Paths

- Single-chip full model: not implemented or selected.
- Replicated host-side decoder: not implemented or selected.
- Full-logit host readback for optimized greedy token-out: rejected and removed
  from the measured free-running path.
- Python token feedback loop for decode: rejected. The trace body consumes a
  persistent 1-wide TT decode input, samples into a persistent tile-width TT
  output buffer, and copies the selected token back to the decode input on
  device.
- Native sampler top-k all-gather under watcher: rejected for this model after
  watcher BRISC assert evidence; replaced by the opt-in composite gather path.
