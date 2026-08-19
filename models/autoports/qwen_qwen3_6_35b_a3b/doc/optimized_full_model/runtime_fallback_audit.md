# Runtime Fallback Audit

The optimized token-out measurement path is fully traced for steady-state
decode and keeps token feedback on device.

## Measured Path

Included:

- replicated BF16 embedding;
- all 40 optimized multichip decoder layers;
- BF16 final RMSNorm;
- BF8 flat 4-way vocab-sharded LM head;
- common on-device greedy sampler;
- device-side token feedback through `tt_out_tok`;
- device-side current-position increment and RoPE lookup;
- persistent cache, page-table, token, position, and RoPE inputs.

Excluded from the measured steady-state loop:

- host argmax;
- full-logits readback;
- Python token feedback;
- per-token host page-table rebuild;
- per-token token buffer refresh;
- per-token position or RoPE refresh;
- per-token host/device synchronization;
- single-chip or replicated decoder fallback.

The only readback in the optimized prompt-128/gen-128 artifact is the optional
terminal validation token read after the measured loop. It is outside the
steady-state decode timing.

## Evidence

| Check | Result | Artifact |
| --- | --- | --- |
| source/contract pytest | `2 passed, 5 skipped` | `logs/pytest_full_model_contract_final.log` |
| synthetic hardware smoke | `7 passed, 2 warnings` | `logs/hardware_smokes_watcher_final.log` |
| throw-on-fallback hardware smoke | `7 passed, 2 warnings` | `logs/synthetic_full_model_no_fallback_smoke_final.log` |
| watcher hardware smoke | `7 passed, 2 warnings` | `logs/hardware_smokes_watcher_final.log` |
| token-out no-readback measurement | final token matches readback baseline | `artifacts/token_out_no_readback_prompt128_gen128_warmed.json` |

The throw-on-fallback run used:
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'` and
`RUN_QWEN36_FULL_MODEL_SMOKE=1`.

The watcher run used:
`TT_METAL_WATCHER=10`,
`TT_METAL_WATCHER_DISABLE_ETH=1`, and
`RUN_QWEN36_FULL_MODEL_SMOKE=1`.

The accepted watcher evidence disables active Ethernet checks because this
p300c host has the active-Ethernet watcher teardown limitation already recorded
by the multichip decoder and full-model stages. No watcher or device failure is
used as passing evidence.
