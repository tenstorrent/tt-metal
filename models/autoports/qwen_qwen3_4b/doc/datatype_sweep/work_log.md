# Qwen3-4B Datatype Sweep Work Log

Base repo commit before stage-owned changes: `b73d429b9fe78319f8f72c1a2ad9e2e4f9543175`.

## Device Preparation

- `tt-smi -ls --local`: found four local Blackhole p150b chips, IDs 0-3.
- `tt-smi -r`: reset all PCI devices and reinitialized boards before the sweep.
- Mesh smoke: opened and closed `ttnn.MeshShape(1, 4)` successfully.
- Hardware regime: `P150_X4` 1x4 TP4 ring with `FABRIC_1D_RING` for CCL readiness runs.

## Implementation Notes

- Added selected precision-config loading in `tt/model.py` and default consumption in `tt/generator.py`.
- Propagated weight dtypes, activation/residual dtype, CCL dtype, KV-cache dtype, logits dtype, and compute fidelities into the measured full-model runtime path.
- Added host contract coverage in `tests/test_full_model_contract.py` for policy-field mapping and `num_layers` override preservation.
- `policy_propagation_check.json` compares `selected_precision_config.json` to the policy summary printed by the measured post-selection default path.

## Refreshed Baseline

- Prefill refresh: top1 93/100, top5 100/100, top100 100/100, log `logs/baseline_bfp4_lofi_bf16kv_bf16ccl_prefill.log`.
- Teacher forcing refresh: top1 94/100, top5 100/100, top100 100/100, TTFT 872.85 ms, traced decode 64.30 t/s/u, log `logs/baseline_bfp4_lofi_bf16kv_bf16ccl_teacher.log`.

## Sweep Decision

Selected `baseline_bfp4_lofi_bf16kv_bf16ccl` because it was the fastest passing traced teacher-forcing candidate.

- `bfp4_lofi_bfp8kv_bfp8ccl`: pass=pass, top1=0.94, top5=1.00, top100=1.00, traced teacher-forcing decode=62.99 t/s/u, TTFT=7150.14 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_lofi_bfp8kv_bf16ccl`: pass=pass, top1=0.93, top5=0.99, top100=1.00, traced teacher-forcing decode=26.42 t/s/u, TTFT=7677.82 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_hifi2_bf16kv_bf16ccl`: pass=pass, top1=0.94, top5=1.00, top100=1.00, traced teacher-forcing decode=23.40 t/s/u, TTFT=4261.78 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp8_hifi2_bf16kv_bf16ccl`: pass=pass, top1=0.98, top5=1.00, top100=1.00, traced teacher-forcing decode=23.02 t/s/u, TTFT=4232.39 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp8_lofi_bf16kv_bf16ccl`: pass=pass, top1=0.98, top5=1.00, top100=1.00, traced teacher-forcing decode=22.86 t/s/u, TTFT=4212.84 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_lofi_bf16kv_bfp8ccl`: pass=pass, top1=0.93, top5=1.00, top100=1.00, traced teacher-forcing decode=19.41 t/s/u, TTFT=5589.06 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.

## Post-Selection Evidence

- Wrote `selected_precision_config.json` and measured token-out through the normal default construction path, no explicit precision override.
- Token-out no-readback: TTFT 332.73 ms, decode 65.78 t/s/u, steady decode 96.91 t/s/u, trace replays 98, readbacks 0.
- Context contract recomputed: `target=40960, supported=40960`; selected KV dtype BF16 did not reduce capacity.
- Non-aligned prompt support preserved by selected BF16 KV/cache layout and AIME24 prompt length 158 evidence.
- Selected qualitative: 6/6 shared prompts passed, log `logs/selected_qualitative.log`.

## Artifacts

- `sweep_results.json`
- `sweep_results.csv`
- `selected_precision_config.json`
- `policy_propagation_check.json`
- `post_selection_token_out_no_readback_benchmark.json`
- `top1_perf_pareto.png`
- `top5_perf_pareto.png`
- `qualitative/qualitative_prompt_format.json`
- `qualitative/qualitative_rendered_prompts.json`
- `qualitative/vllm_qualitative_outputs.json`
- `qualitative/qualitative_verdict.json`
- `logs/*.log`

## Verification

- `python -m py_compile models/autoports/qwen_qwen3_4b/tt/model.py models/autoports/qwen_qwen3_4b/tt/generator.py models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`: passed.
- `pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short`: 11 passed, 2 nanobind ref-leak warnings at process exit.
- Stage review: clean-pass from subagent `019f2533-e829-7ff2-a7be-e3d5f95fce84`; no required work. Residual risks noted by review: nanobind ref-leak shutdown warnings, firmware newer-than-tested notices, and the one-reference AIME24 sweep scope already documented in `README.md`.
- Local commit SHA: pending at review time; recorded in final handoff after the local commit is created.
