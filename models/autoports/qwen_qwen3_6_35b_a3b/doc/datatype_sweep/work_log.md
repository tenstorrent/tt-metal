# Datatype Sweep Work Log

## Provenance

- Model: `Qwen/Qwen3.6-35B-A3B`
- Autoport directory: `models/autoports/qwen_qwen3_6_35b_a3b`
- Stage base SHA: `13bef365ad22d21aef9ddb995f324a881ebd36e1`
- Final local commit SHA: recorded after commit; the commit cannot contain its own final SHA.
- Hardware: 4 local Blackhole P300C devices, mesh `[2, 2]`, `FABRIC_1D_RING`
- Common environment: `TT_METAL_WATCHER_DISABLE_ETH=1`, `TT_READINESS_TRACE_REGION_SIZE=64000000`

## Device Use

- Ran `timeout 60 tt-smi -ls --local | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/tt_smi_initial.log`.
- Ran a 2x2 mesh-open smoke with `P300C` and `FABRIC_1D_RING`; log: `logs/mesh_open_smoke.log`.
- Serialized all hardware runs. No TT commands were run in parallel.

## Implementation

- Added `tt/precision_config.py` for default/env/explicit precision config loading and dtype/fidelity parsing.
- Plumbed weight-group dtypes, compute fidelities, CCL dtype, KV-cache dtype, linear-state dtype, logits dtype, and sampling dtype through `tt/model.py`, `tt/generator.py`, `tt/optimized_decoder.py`, and `tt/multichip_decoder.py`.
- Added runtime policy introspection with `QwenFullModel.describe_precision_policy()`.
- Added `doc/datatype_sweep/scripts/evaluate_candidate.py` for full-model candidate measurement.
- Added `doc/datatype_sweep/scripts/build_artifacts.py` for aggregate results, CSV, plots, selected config generation, and quality-gate overrides.
- Updated `models/common/readiness_check/check_degenerate_output.py` to catch repeated punctuation and raw token-id runs.
- Updated the optimized-full-model token-out benchmark to record runtime precision policy, command, and env in its JSON payload.

## Checks

- JSON parse: `doc/context_contract.json`, `sweep_results.json`, `selected_precision_config.json`, and selected token-out JSON.
- `./python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_35b_a3b/tt/precision_config.py models/autoports/qwen_qwen3_6_35b_a3b/tt/model.py models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py models/autoports/qwen_qwen3_6_35b_a3b/tt/optimized_decoder.py models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/scripts/evaluate_candidate.py models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/scripts/build_artifacts.py models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/measure_token_out_no_readback.py`
- `./python_env/bin/python -m pytest -q --confcutdir=models/autoports/qwen_qwen3_6_35b_a3b models/autoports/qwen_qwen3_6_35b_a3b/tests/test_full_model.py -k 'not FULL_MODEL_SMOKE'`
- Selected non-aligned smoke: `artifacts/selected_non_aligned_prompt_smoke.json`

## Candidate Command

Every full-model candidate used this command form, with `<cfg>` set to the config id:

```bash
timeout 7200 env TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/scripts/evaluate_candidate.py \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --reference models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/artifacts/aime24_chat_100.refpt \
  --config-id <cfg> \
  --precision-config models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/configs/<cfg>.json \
  --mesh-device P300C \
  --fabric-config FABRIC_1D_RING \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/<cfg>_result.json \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/<cfg>_evaluate.log
```

Candidate ids:

- `baseline_default`
- `bfp8_hifi2_all`
- `bfp8_lofi_all`
- `kv_bf8_default`
- `ccl_bf8_default`
- `routed_all_bfp4_lofi`
- `routed_all_bfp4_hifi2`
- `shared_moe_bfp4_lofi`
- `shared_moe_bfp4_hifi2`

Each result JSON embeds its exact `command`, `env`, `hardware`, and `mesh` fields.

## Candidate Results

| Config | Status | TF top-1 | TF top-5 | TF top-100 | TTFT ms | TF decode t/s/u | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `shared_moe_bfp4_lofi` | fail_quality | 0.95 | 1.00 | 1.00 | 9126.5 | 16.3838 | no |
| `baseline_default` | pass | 0.99 | 1.00 | 1.00 | 9079.1 | 16.3785 | yes |
| `kv_bf8_default` | pass | 0.98 | 1.00 | 1.00 | 11124.4 | 16.3721 | no |
| `bfp8_lofi_all` | pass | 0.96 | 1.00 | 1.00 | 8896.1 | 16.3720 | no |
| `routed_all_bfp4_hifi2` | pass | 0.96 | 1.00 | 1.00 | 8620.9 | 16.3661 | no |
| `shared_moe_bfp4_hifi2` | pass | 0.95 | 1.00 | 1.00 | 8569.5 | 16.3636 | no |
| `bfp8_hifi2_all` | pass | 0.95 | 1.00 | 1.00 | 8846.3 | 16.3612 | no |
| `routed_all_bfp4_lofi` | pass | 0.96 | 1.00 | 1.00 | 9082.0 | 16.3598 | no |
| `ccl_bf8_default` | pass | 0.97 | 1.00 | 1.00 | 9063.1 | 16.1627 | no |

## Selection

`baseline_default` is the fastest evaluated overall passing candidate after the quality gate. It passed the 0.90 top-1 and 0.98 top-5 gate with top-1 0.99, top-5 1.00, top-100 1.00, TTFT 9079.1 ms, and traced teacher-forcing decode 16.3785 t/s/u.

`shared_moe_bfp4_lofi` was faster at 16.3838 t/s/u, but it failed stage review and the patched qualitative repetition gate after producing repeated punctuation and token-id runs. It is preserved as rejected evidence and excluded from final selection by `quality_gate_overrides.json`.

BFP4 coverage:

- Routed MoE has both `routed_all_bfp4_lofi` and `routed_all_bfp4_hifi2`.
- Shared MoE has both `shared_moe_bfp4_lofi` and `shared_moe_bfp4_hifi2`.

## Artifact Commands

```bash
./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/scripts/build_artifacts.py \
  --sweep-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/build_artifacts_after_rejected_metadata_fix.log
```

```bash
env -u QWEN36_PRECISION_CONFIG ./python_env/bin/python - <<'PY' 2>&1 \
  | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/selected_precision_default_load_check_after_quality_gate.log
# Inline loader check wrote artifacts/selected_precision_default_load_check.json.
PY
```

```bash
timeout 7200 env -u QWEN36_PRECISION_CONFIG TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/scripts/measure_token_out_no_readback.py \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --prompt-len 128 \
  --max-new-tokens 128 \
  --mesh-device P300C \
  --fabric-config FABRIC_1D_RING \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/token_out_no_readback_selected_prompt128_gen128_warmed_after_quality_gate.log
```

```bash
timeout 7200 env -u QWEN36_PRECISION_CONFIG TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=64000000 \
  ./python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/scripts/run_qualitative_chat_suite.py \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/qualitative_chat_suite_64 \
  --max-new-tokens 64 \
  --skip-hf \
  --mesh-device P300C \
  --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/run_qualitative_chat_suite_64_baseline_selected.log
```

```bash
./python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/qualitative_chat_suite_64/vllm_qualitative_outputs.json \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/qualitative_chat_suite_64/tt_qualitative_outputs.json \
  --scope vllm \
  --missing-artifacts critical \
  --json models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/artifacts/qualitative_chat_suite_64/degenerate_output_report.json \
  2>&1 | tee models/autoports/qwen_qwen3_6_35b_a3b/doc/datatype_sweep/logs/check_degenerate_output_qualitative_chat_suite_64_baseline_selected.log
```

## Post-Selection Token-Out

- Artifact: `artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json`
- TTFT: 5841.2 ms
- Decode replay: 17.4361 t/s/u
- Decode including trace capture: 16.7702 t/s/u
- Runtime source: `doc/datatype_sweep/selected_precision_config.json`
- Steady-state token readbacks: 0

## Context Contract

- Recomputed `doc/context_contract.json` for selected BF16 KV-cache policy.
- Added `artifacts/context_contract_recompute_selected_precision.json`.
- Advertised context remains 262144 tokens.
- No prefill or decode capability reduction.
- `kv_bf8_default` is recorded as evaluated and rejected; it was not selected, so no BF8-KV advertised capability applies.

## Qualitative Verdict

- Prompt format: HF tokenizer chat template with `add_generation_prompt=True`.
- HF control: reused from `doc/full_model/artifacts/qualitative_chat_suite_64/hf_qualitative_outputs.json` into this stage's final artifact directory.
- TT outputs: regenerated with final `baseline_default` selected config and traced decode for all 6 prompts.
- Patched degenerate checker exit code: 0, no findings.
- Rejected candidate evidence: `artifacts/qualitative_chat_suite_64_rejected_shared_moe_bfp4_lofi/rejected_shared_moe_bfp4_lofi_repetition_report.json` records the repeated punctuation and token-id runs that caused `shared_moe_bfp4_lofi` to fail quality.

## Stage Review

- Initial review `01a01b13-6a3a-7d73-8f81-103b22af1c95`: `more-work-needed`.
- Initial fixes: preserved and rejected `shared_moe_bfp4_lofi` qualitative evidence, patched `check_degenerate_output.py`, regenerated final selected config as `baseline_default`, reran token-out, qualitative, non-aligned, context, and docs.
- Rereview `01a01b29-28fc-7aa0-8d75-e5fce6eae54f`: `more-work-needed`.
- Rereview fixes: regenerated rejected report against the rejected directory, replaced its verdict with `rejected`, and updated selected-config metadata to include the qualitative quality gate.
- Final rereview `01a01b2f-2403-7461-b92c-e1ddaaec1548`: `clean-pass`.

## Limitations

- Candidate ranking uses one full-model measurement per config.
- Teacher-forcing decode is the performance ranking source by requirement; eager and token-out values are not used for Pareto ranking.
- The final selection applies the stage qualitative gate in addition to top-1/top-5 accuracy. Without that quality gate, `shared_moe_bfp4_lofi` would be the fastest traced teacher-forcing point, but it is not an overall passing config.
- Post-selection token-out is recorded separately and should be used for later serving-style comparisons.
