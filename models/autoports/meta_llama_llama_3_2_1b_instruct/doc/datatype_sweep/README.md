# Datatype Sweep

Model: `meta-llama/Llama-3.2-1B-Instruct`

Selected config: `cfg08_bfp8_weights_bfp8_kv_bf16_ccl`. This is the fastest evaluated config that passes the full-model gate using trace-verified teacher-forcing decode performance.

## Selected Result

| Metric | Value |
| --- | ---: |
| Top-1 threshold | 90% |
| Top-5 threshold | 98% |
| Selected top-1 | 96% (96/100) |
| Selected top-5 | 100% (100/100) |
| Selected top-100 | 100% (100/100) |
| Teacher-forcing TTFT | 183.03 ms |
| Trace-verified teacher-forcing decode | 92.68 t/s/u |
| Post-selection token-out TTFT | 1407.02 ms |
| Post-selection token-out decode | 146.75 t/s/u |
| Token-out measured-loop readbacks | 0 |

Selected policy:

- Attention weights: `bfloat8_b`.
- MLP gate/up and down weights: `bfloat8_b`.
- KV cache: `bfloat8_b`.
- Activation, residual, RMSNorm, logits, and sampler input assumptions: `bfloat16`.
- CCL all-gather and reduce-scatter payloads: `bfloat16` with persistent buffers.
- Compute fidelity: HiFi2 fp16 accumulate for decode matmuls/norms/LM head, HiFi4 fp32 dest for prefill SDPA.
- Layer exceptions: none; all 16 decoder layers use the selected policy.

`selected_precision_config.json` is consumed by default by `build_generator` and by `Llama32FullModel.from_pretrained` when the autoport `.ttnn_cache` path is used. Set `MD_LLAMA32_PRECISION_CONFIG=none` to return to the built-in optimized baseline.

## Baseline Refresh

The built-in optimized baseline was refreshed before writing `selected_precision_config.json`, using the AIME24 chat-template `gen_len=100` reference:

| Check | Top-1 | Top-5 | Top-100 | TTFT ms | Decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official `run_prefill_check` | 88% | 100% | 100% | n/a | n/a |
| Official traced `run_teacher_forcing` | 86% | 100% | 100% | 245.25 | 98.50 |

## Sweep Results

| Config | Status | Top-1 | Top-5 | Top-100 | TTFT ms | Traced TF decode t/s/u | Attn W | MLP W | KV | CCL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `cfg06_bf16_attn_bfp4_mlp_bfp8_kv_ccl` | fail | 86% | 100% | 100% | 255.06 | 93.39 | bfloat16 | bfloat4_b/bfloat4_b | bfloat8_b | bfloat8_b |
| `cfg08_bfp8_weights_bfp8_kv_bf16_ccl` | pass | 96% | 100% | 100% | 183.03 | 92.68 | bfloat8_b | bfloat8_b/bfloat8_b | bfloat8_b | bfloat16 |
| `cfg04_bfp8_weights_bf16_kv_ccl` | pass | 96% | 100% | 100% | 181.60 | 92.13 | bfloat8_b | bfloat8_b/bfloat8_b | bfloat16 | bfloat16 |
| `cfg00_optimized_default` | fail | 86% | 100% | 100% | 272.94 | 90.97 | bfloat8_b | bfloat4_b/bfloat4_b | bfloat8_b | bfloat8_b |
| `cfg09_bfp8_weights_bf16_kv_bfp8_ccl` | pass | 97% | 100% | 100% | 258.87 | 90.95 | bfloat8_b | bfloat8_b/bfloat8_b | bfloat16 | bfloat8_b |
| `cfg05_bfp8_attn_bfp4_gateup_bf16_down_bfp8_kv_ccl` | fail | 88% | 100% | 100% | 249.39 | 88.92 | bfloat8_b | bfloat4_b/bfloat16 | bfloat8_b | bfloat8_b |
| `cfg02_bf16_weights_bfp8_kv_ccl` | pass | 97% | 100% | 100% | 258.63 | 80.63 | bfloat16 | bfloat16/bfloat16 | bfloat8_b | bfloat8_b |
| `cfg07_bfp8_attn_bf16_mlp_bfp8_kv_ccl` | pass | 95% | 100% | 100% | 251.88 | 79.07 | bfloat8_b | bfloat16/bfloat16 | bfloat8_b | bfloat8_b |
| `cfg03_bfp8_weights_bfp8_kv_ccl` | pass | 97% | 100% | 100% | 1632.63 | 41.21 | bfloat8_b | bfloat8_b/bfloat8_b | bfloat8_b | bfloat8_b |
| `cfg01_all_bf16` | pass | 97% | 100% | 100% | 5201.19 | 23.03 | bfloat16 | bfloat16/bfloat16 | bfloat16 | bfloat16 |

## Pareto Interpretation

`top1_perf_pareto.png` and `top5_perf_pareto.png` plot all evaluated full-model configs using trace-verified teacher-forcing decode t/s/u. The selected point is red and the vertical dotted line is the accuracy gate.

For top-1, `cfg08` is the fastest evaluated passing point. `cfg06` is slightly faster but fails top-1 at 86/100, and `cfg00`/`cfg05` also fail. `cfg04` is close to `cfg08`, but `cfg08` is the measured fastest passing point and keeps the lower-memory BFP8 KV cache.

For top-5, every evaluated config reaches 100/100, so top-1 is the binding accuracy gate.

## Rejected Configs

- `cfg00_optimized_default`: failed top-1 at 86/100; BFP4 MLP is too aggressive for the gate.
- `cfg05_bfp8_attn_bfp4_gateup_bf16_down_bfp8_kv_ccl`: failed top-1 at 88/100; restoring only FF2/down is insufficient.
- `cfg06_bf16_attn_bfp4_mlp_bfp8_kv_ccl`: failed top-1 at 86/100; restoring attention alone is insufficient.
- Passing BF16-heavy configs were rejected because they were slower than `cfg08` under traced teacher forcing.

## Commands And Artifacts

Key commands are in `work_log.md`. Raw per-candidate logs and JSON live under `runs/<config_id>/`.

Required artifacts:

- `sweep_results.json`
- `sweep_results.csv`
- `selected_precision_config.json`
- `top1_perf_pareto.png`
- `top5_perf_pareto.png`
- `post_selection_token_out/token_out_no_readback_perf.json`
- `propagation_check.json`

## Limitations

- This sweep uses one AIME24 chat-template reference with 100 generated tokens, matching the readiness gate requested for this stage.
- The post-selection token-out benchmark is separate from teacher forcing and should be used for later serving comparisons when available.
- No vLLM integration work was started. No `generator_vllm.py` exists in this autoport yet; the selected artifact is ready for that later path to consume through `build_generator` or `Llama32FullModel.from_pretrained`.
