# Datatype Sweep Work Log

Date: 2026-06-15.

Scope: `$datatype-sweep` for `meta-llama/Llama-3.1-8B-Instruct` optimized full model under `models/autoports/meta_llama_llama_3_1_8b_instruct`. vLLM integration was not started.

## Setup

- Branch: `agentic-research/experiment-17-llama31-8b`
- Commit: `86f8bc022e6d526d9766539c6ea50137cabec799`
- Hardware: 8 Wormhole chips visible via `tt-smi`, T3K `1x8`, `FABRIC_1D_RING`
- Accuracy thresholds: top-1 >= 0.90, top-5 >= 0.98; top-100 expectation 1.0
- Reference: `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/aime24_chat_template_100_top100.refpt`, AIME24 prompt source, chat template, 100 generated tokens, top-k 100, prompt length 184

## Implementation Notes

Added selected precision config loading to the normal full-model construction path:

- `tt/model.py` reads `doc/datatype_sweep/selected_precision_config.json` by default when present, or `LLAMA31_8B_PRECISION_CONFIG` when set.
- `tt/generator.py::build_generator()` forwards `precision_config_path` into `from_pretrained()`.
- `tt/multichip_decoder.py` now preserves the supplied policy name instead of replacing it with the class default.
- `LLAMA31_8B_PRECISION_CONFIG=baseline` returns to the hardcoded code-default policy.

Propagation check: `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/runs/selected_config_propagation_check.log` shows default `build_generator(..., override_num_layers=1)` loaded `selected_precision_config.json` and reported BF4 attention/MLP, BF8 activation/KV, BF8 LM head. No vLLM adapter file exists yet.

## Baseline Refresh

Generated a fresh AIME24 chat-template reference with 100 generated tokens. Baseline prefill top-1/top-5/top-100: 0.930/1.000/1.000. Baseline traced teacher forcing top-1/top-5/top-100: 0.900/1.000/1.000, TTFT 646.52 ms, decode 49.49 t/s/u.

## Candidate Results

| `baseline_default` | pass | 0.900 | 1.000 | 1.000 | 646.52 | 49.49 | selected |
| `safe_bfp8_bf16_act_hifi2` | pass | 0.960 | 1.000 | 1.000 | 790.51 | 27.84 | passing but slower |
| `bf16_activation_current_weights` | pass | 0.900 | 1.000 | 1.000 | 615.08 | 48.52 | passing but slower |
| `bf16_kv_current_weights` | pass | 0.910 | 1.000 | 1.000 | 1055.82 | 25.31 | passing but slower |
| `bfp4_kv_current_weights` | fail | 0.830 | 0.990 | 1.000 | 1066.06 | 23.88 | top1 0.830 < 0.90 |
| `bfp4_lm_head_current_weights` | fail | 0.780 | 0.970 | 1.000 | 628.45 | 39.80 | top1 0.780 < 0.90; top5 0.970 < 0.98 |
| `first_last_bfp8_guardrails` | pass | 0.900 | 1.000 | 1.000 | 672.67 | 29.08 | passing but slower |

Rejected configs:

- `safe_bfp8_bf16_act_hifi2`: passes accuracy but is slower than selected.
- `bf16_activation_current_weights`: passes accuracy but is slower than selected.
- `bf16_kv_current_weights`: passes accuracy but is slower than selected.
- `first_last_bfp8_guardrails`: passes accuracy but is slower than selected.
- `bfp4_kv_current_weights`: fails top-1 (`0.830 < 0.90`).
- `bfp4_lm_head_current_weights`: fails top-1 (`0.780 < 0.90`) and top-5 (`0.970 < 0.98`).

## Selection

Selected `baseline_default`, written as `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/selected_precision_config.json`. It is the fastest passing full-model config by traced teacher-forcing decode performance: 49.49 t/s/u at top-1 0.900 and top-5 1.000.

Compute fidelity remains: MLP LoFi for BFP4 MLP weights; attention and LM head HiFi2; BF16 norms. This matches the optimized decoder evidence and the fastest passing full-model result.

## Post-Selection Token-Out

Command:

```bash
timeout 7200s python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt --mesh-device T3K --fabric-config FABRIC_1D_RING --max-new-tokens 128 --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/post_selection_token_out_trace_evidence.json
```

Result: TTFT 635.27 ms, steady no-readback replay 70.55 t/s/u (14.17 ms/token), 0 sampled-token readbacks, model trace captured `True`.

## Qualitative Check

Because selected top-1 is exactly at threshold, ran selected-config autoregressive story generation and degeneracy check. `check_degenerate_output.py` reported no degenerate output. Informational HF/TT token agreement: 30/128. Adjacent duplication 0.0; trigram loop fraction 0.0297.

## Artifacts

- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/sweep_results.json`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/sweep_results.csv`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/selected_precision_config.json`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/top1_perf_pareto.png`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/top5_perf_pareto.png`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/post_selection_token_out_trace_evidence.json`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/autoregressive_degenerate_report.json`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/runs`

## Limitations

The sweep uses the readiness AIME24 chat-template reference, not a broad benchmark suite. BFP4 KV and BFP4 LM-head were evaluated as aggressive lower-precision wins and failed accuracy. Separate attention QKV/WO or gate/up split dtypes are not yet supported by the full-model constructor because the current multichip modules bind those pairs to one dtype each.
