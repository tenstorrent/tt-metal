# Qwen3.6-35B-A3B Datatype Sweep

## Scope

This stage starts from the completed optimized full model for `Qwen/Qwen3.6-35B-A3B` and selects the fastest full-model precision policy that preserves the AIME24 chat-template readiness gate and the stage qualitative gate. No vLLM integration work is included.

The final selected policy is `baseline_default`. It is the default repo-local precision config at `doc/datatype_sweep/selected_precision_config.json`, consumed by `tt/generator.py::build_generator` through `tt/precision_config.py::load_precision_config` when no explicit precision config and no `QWEN36_PRECISION_CONFIG` override are supplied.

`shared_moe_bfp4_lofi` was the fastest traced teacher-forcing accuracy-pass candidate, but stage review found prompt-correct qualitative collapse on prompts 1 and 4. The degenerate-output checker was extended to catch repeated punctuation and token-id runs, and `shared_moe_bfp4_lofi` is marked `fail_quality` in `quality_gate_overrides.json`, `sweep_results.json`, and `sweep_results.csv`.

## Gates

- Reference: `doc/optimized_full_model/artifacts/aime24_chat_100.refpt`
- Prompt format: HF tokenizer chat template, 161 prompt tokens, 100 generated reference tokens
- Accuracy gate: traced teacher-forcing top-1 >= 0.90 and top-5 >= 0.98
- Quality gate: prompt-correct qualitative suite plus degenerate-output check over generated text and raw token ids
- Ranking metric: trace-verified teacher-forcing decode t/s/u among overall passing candidates
- Hardware: 4 local Blackhole P300C devices, mesh shape `[2, 2]`, `FABRIC_1D_RING`
- Common environment: `TT_METAL_WATCHER_DISABLE_ETH=1`, `TT_READINESS_TRACE_REGION_SIZE=64000000`

## Selected Policy

- Config id: `baseline_default`
- Weight groups: embedding BF16, norms BF16, attention BF8, linear attention BF8, router BF16, shared MoE BF8, routed MoE BF8 on linear-attention layers and BF4 on full-attention layers, LM head BF8
- Layer exceptions: none
- Compute fidelities: TTNN default for all material groups
- Activation/residual dtype: BF16
- CCL dtype: BF16
- KV-cache dtype: BF16
- Linear recurrent-state dtype: BF16
- Logits dtype: BF16
- Sampling dtype assumption: uint32 token buffers with greedy top-k1 device sampling

Consumption evidence:

- `artifacts/baseline_default_result.json` records runtime policy summary for the measured selected candidate.
- `artifacts/selected_precision_default_load_check.json` proves the default loader resolves `selected_precision_config.json` with no env override.
- `artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json` records the post-selection benchmark runtime source as `doc/datatype_sweep/selected_precision_config.json`.

## Results

| Config | Status | TF top-1 | TF top-5 | TF top-100 | TTFT ms | TF decode t/s/u | Prefill top-1/top-5 | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `shared_moe_bfp4_lofi` | fail_quality | 0.95 | 1.00 | 1.00 | 9126.5 | 16.3838 | 0.96/1.00 | no |
| `baseline_default` | pass | 0.99 | 1.00 | 1.00 | 9079.1 | 16.3785 | 0.96/1.00 | yes |
| `kv_bf8_default` | pass | 0.98 | 1.00 | 1.00 | 11124.4 | 16.3721 | 0.98/1.00 | no |
| `bfp8_lofi_all` | pass | 0.96 | 1.00 | 1.00 | 8896.1 | 16.3720 | 0.95/1.00 | no |
| `routed_all_bfp4_hifi2` | pass | 0.96 | 1.00 | 1.00 | 8620.9 | 16.3661 | 0.96/1.00 | no |
| `shared_moe_bfp4_hifi2` | pass | 0.95 | 1.00 | 1.00 | 8569.5 | 16.3636 | 0.96/1.00 | no |
| `bfp8_hifi2_all` | pass | 0.95 | 1.00 | 1.00 | 8846.3 | 16.3612 | 0.99/1.00 | no |
| `routed_all_bfp4_lofi` | pass | 0.96 | 1.00 | 1.00 | 9082.0 | 16.3598 | 0.96/1.00 | no |
| `ccl_bf8_default` | pass | 0.97 | 1.00 | 1.00 | 9063.1 | 16.1627 | 0.98/1.00 | no |

Full machine-readable results are in `sweep_results.json` and `sweep_results.csv`. Each row includes config id, dtype policy, compute-fidelity policy, top-1/top-5/top-100, TTFT, traced teacher-forcing decode t/s/u, command, hardware, mesh, measurement regime, trace counters, runtime policy summary, quality gate, and pass/fail status.

## Pareto Interpretation

- `top1_perf_pareto.png` marks `baseline_default` as the selected point after the quality gate. `shared_moe_bfp4_lofi` remains visible as the fastest measured teacher-forcing point but is not an overall passing candidate.
- `top5_perf_pareto.png` has all evaluated configs at 100% top-5; throughput separates the points, with the selected point marked red.
- The vertical dotted line marks the minimum allowed accuracy: 90% top-1 in `top1_perf_pareto.png`, 98% top-5 in `top5_perf_pareto.png`.

## Rejected Configs

- `shared_moe_bfp4_lofi`: fastest traced teacher-forcing accuracy-pass candidate, but rejected by quality gate. It produced long repeated `!` runs and raw token-id 0 runs on prompts 1 and 4. Evidence: `artifacts/qualitative_chat_suite_64_rejected_shared_moe_bfp4_lofi/rejected_shared_moe_bfp4_lofi_repetition_report.json`.
- `kv_bf8_default`: passed and would reduce KV memory, but traced teacher-forcing decode was slower than selected at 16.3721 t/s/u and TTFT regressed to 11124.4 ms.
- `bfp8_lofi_all` and `bfp8_hifi2_all`: passed, but both were slower than selected.
- `routed_all_bfp4_lofi` and `routed_all_bfp4_hifi2`: both passed, satisfying the BFP4 LoFi/HiFi2 coverage requirement for routed MoE, but were slower than selected.
- `shared_moe_bfp4_hifi2`: passed and provides the paired BFP4 fidelity coverage for shared MoE, but was slower than selected.
- `ccl_bf8_default`: passed accuracy but was the slowest traced teacher-forcing decode point.

## Post-Selection Token-Out

The final selected config was rerun through the normal default construction path using the same warmed token-out no-readback benchmark from optimized full model:

- Artifact: `artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json`
- Prompt length: 128
- Generated tokens: 128
- TTFT: 5841.2 ms
- Decode replay: 17.4361 t/s/u
- Decode including trace capture: 16.7702 t/s/u
- Steady-state token readbacks: 0
- Runtime source: `doc/datatype_sweep/selected_precision_config.json`

Later reports should use this post-selection token-out result when comparing serving-style decode performance.

## Context And Prompt Length

The selected KV-cache dtype is BF16, matching optimized full model. `doc/context_contract.json` was recomputed for the selected policy and still advertises 262144 context tokens. Runtime cache/state capacity is unchanged:

- Selected per-device transformed weights at context 262144: 15.9851 GB
- Runtime state excluding weights: 2.7682 GB
- Weight plus runtime state: 18.7533 GB
- Capacity reduction: false

`kv_bf8_default` was evaluated as the KV-memory-changing candidate and rejected on performance. No advertised capability reduction was applied.

Non-aligned prompt support was rerun for the selected config:

- Artifact: `artifacts/selected_non_aligned_prompt_smoke.json`
- Prompt length: 5
- Prefill shape: `[1, 5, 128]`
- Decode shape: `[1, 128]`
- Runtime policy confirmed baseline shared-MoE BF8, routed-MoE BF4 on full-attention layers, and KV-cache BF16.

## Qualitative Check

The final selected config regenerated the shared chat-template qualitative suite:

- Prompt metadata: `artifacts/qualitative_chat_suite_64/qualitative_prompt_format.json`
- HF control reused from full-model same-prompt run: `artifacts/qualitative_chat_suite_64/hf_qualitative_outputs.json`
- Selected TT outputs: `artifacts/qualitative_chat_suite_64/tt_qualitative_outputs.json`
- Degenerate-output report: `artifacts/qualitative_chat_suite_64/degenerate_output_report.json`

The patched degenerate-output checker exited 0 with no findings for `baseline_default`. The rejected `shared_moe_bfp4_lofi` qualitative evidence is preserved under `artifacts/qualitative_chat_suite_64_rejected_shared_moe_bfp4_lofi/`.

## Artifacts

- `sweep_results.json`
- `sweep_results.csv`
- `selected_precision_config.json`
- `quality_gate_overrides.json`
- `top1_perf_pareto.png`
- `top5_perf_pareto.png`
- `artifacts/*_result.json`
- `artifacts/context_contract_recompute_selected_precision.json`
- `artifacts/selected_precision_default_load_check.json`
- `artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json`
- `artifacts/selected_non_aligned_prompt_smoke.json`
- `artifacts/qualitative_chat_suite_64/`
- `artifacts/qualitative_chat_suite_64_rejected_shared_moe_bfp4_lofi/`
- `logs/`

## Stage Review

- Initial review `01a01b13-6a3a-7d73-8f81-103b22af1c95`: `more-work-needed`; fixed by rejecting `shared_moe_bfp4_lofi`, adding repeated punctuation/token-id checks, and selecting `baseline_default`.
- Rereview `01a01b29-28fc-7aa0-8d75-e5fce6eae54f`: `more-work-needed`; fixed stale rejected-artifact paths, rejected verdict text, and selected-config quality-gate metadata.
- Final rereview `01a01b2f-2403-7461-b92c-e1ddaaec1548`: `clean-pass`.

## Limitations

- Candidate ranking uses one full-model accuracy/performance measurement per config.
- Teacher-forcing decode is the performance ranking source by requirement; eager and token-out values are not used for Pareto ranking.
- The final selection applies the stage qualitative gate in addition to top-1/top-5 accuracy. Without that quality gate, `shared_moe_bfp4_lofi` would be the fastest traced teacher-forcing point, but it is not an overall passing config.
- The HF qualitative control was reused from the full-model stage because it is the same model, tokenizer chat template, prompt suite, and 64-token generation regime; final selected TT outputs were regenerated for this stage.
