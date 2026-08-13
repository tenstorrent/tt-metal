# Mistral Small 24B datatype sweep

## Result

The selected policy is `bfp4_lofi_bfp8kv_bf16ccl`. It is the fastest evaluated policy that passes the full-model accuracy and capability gates when ranked by the internal trace-verified teacher-forcing decode interval.

| Metric | Selected result | Gate |
|---|---:|---:|
| AIME24 chat-template top-1, 100 tokens | 97/100 | >= 90/100 |
| top-5 | 100/100 | >= 98/100 |
| top-100 | 100/100 | 100/100 |
| traced teacher-forcing decode, median of two | 54.212248 t/s/u | ranking metric |
| teacher-forcing TTFT, median of two | 226.993467 ms | reported, not ranked |
| post-selection warmed token-out no-readback | 55.930356 t/s/u | separate deployment metric |
| post-selection token-out TTFT | 57.294177, 56.826756 ms | separate deployment metric |

The selected policy keeps BFP4_B attention and MLP weights with LoFi dense matmuls, BF16 embedding/norm/LM-head weights, BF16 activations and residuals, BFP8_B paged KV cache, BF16 decode/prefill CCL payloads and workspace, HiFi4 SDPA, HiFi2 LM head, and BF16 logits/sampling tensors. There are no layer exceptions. The complete policy is in `selected_precision_config.json`.

## Baseline and measurement regime

The readiness reference is the main AIME24 chat-template artifact `../full_model/artifacts/aime24_chat_100.refpt` (SHA256 `e88a9c2fe1d59448231e5edc4260f306328e4e4fdeef878d05166d2e4d9bbbc9`). The baseline optimized full model was refreshed for both prefill and split traced teacher-forcing decode with exactly 100 generated tokens:

- Prefill: 99/100 top-1, 100/100 top-5 and top-100 (`logs/baseline_prefill.log`).
- Original baseline traced decode: 97/100, 100/100, 100/100; 225.74 ms TTFT and 52.9074 internal traced t/s/u (`logs/baseline_teacher_forcing.log`).
- Policy-backed repeated baseline: 97/100, 100/100, 100/100; median 229.367625 ms TTFT and 52.829349 traced t/s/u (`logs/baseline_policy_teacher_forcing*.log`).

All Pareto ranking uses `last_generate_stats["traced_decode_t/s/u"]`, measured strictly inside the captured trace replay interval. The readiness process-level number includes setup/readback and is not used for selection. Hardware was four Blackhole p300c devices (firmware 19.9.0) in a 1x4 TP mesh with `FABRIC_1D`.

Representative full-model command (replace the candidate path):

```bash
MISTRAL_SMALL_24B_PRECISION_CONFIG=models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/candidates/bfp4_lofi_bfp8kv_bf16ccl.json \
HF_HUB_OFFLINE=1 TT_LOGGER_LEVEL=info \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --reference models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D --trace-region-size 200000000
```

Exact per-candidate commands, policies, measurements, hardware, pass/fail fields, evidence logs, branch, measurement base commit (`d182c2fe795610b7622205b4a06e98457b2a6e93`), and live datatype-plumbing worktree state are normalized in `sweep_results.json` and `sweep_results.csv`.

## Candidates and rejection evidence

| Config | Top-1 / top-5 | Traced t/s/u | Outcome |
|---|---:|---:|---|
| `bfp4_lofi_bfp8kv_bf16ccl` | 97% / 100% | **54.212248** | selected |
| `bfp4_lofi_bf16kv_bfp8ccl` | 97% / 100% | 53.326028 | rejected: only 18,304-token physical context ceiling |
| `baseline_bfp4_lofi_bfp8kv_bfp8ccl` | 97% / 100% | 52.829349 | rejected: slower in repeated TF and matched token-out |
| `bfp4_lofi_bfp8act_bfp8kv_bfp8ccl` | 98% / 100% | 51.600778 | rejected: lower-precision matmul inputs are slower |
| `bfp8_lofi_bfp8kv_bfp8ccl` | 98% / 100% | 38.339460 | rejected: legal block-8 BFP8 weights are slower |
| `bfp4_hifi2_bfp8kv_bfp8ccl` | 97% / 100% | 36.609832 | rejected: higher dense fidelity is slower |
| `bfp8_hifi2_bfp8kv_bfp8ccl` | 99% / 100% | 35.950609 | rejected: slower |

Every material BFP4 group (attention QKV/WO, MLP gate/up, and MLP down) is present in the baseline and selected BFP4+LoFi candidates. The matched BFP4+HiFi2 candidate isolates fidelity. BFP8 weight policies required the legal block-8 MLP geometry after block-16 exceeded L1; both LoFi and HiFi2 were then measured as runnable full-model policies, so no runtime blocker or autofix exemption is being claimed.

## Pareto interpretation

`top1_perf_pareto.png` and `top5_perf_pareto.png` plot every evaluated full-model policy, connect the accuracy/performance Pareto frontier, mark the selected point in red, and draw the minimum accuracy as a vertical dotted line. All policies pass the numerical top-1/top-5 gate; the BF16-KV point fails the independent capability gate. Because every candidate has 100% top-5, the top-5 frontier reduces to the fastest point at that accuracy.

## Runtime policy consumption

`tt/precision_policy.py` parses the selected artifact. `tt.generator.build_generator` loads an explicit `MISTRAL_SMALL_24B_PRECISION_CONFIG`, or automatically loads this directory's `selected_precision_config.json` when the variable is absent. It passes the policy through `MistralSmall24BFullModel.from_state_dict` into every decoder layer and the sampling path. Runtime fields consumed by the measured path include all weight groups, dense/SDPA/LM-head fidelities, attention and MLP inputs, residuals, decode/prefill CCL payloads, workspace, KV cache, logits, sampling parameters, token indices, layer exceptions, context length, and program geometry.

Full-model generator hardware runs print `PRECISION_POLICY_RUNTIME_SUMMARY`, including the actual first/last-layer values. `logs/policy_one_layer_smoke.log` proves candidate parsing and propagation. `logs/selected_default_non_aligned.log` runs with the override variable unset, proves automatic selected-policy loading, reports BF16 CCL/BFP4-LoFi/BFP8-KV from the constructed runtime, and passes 7-token non-aligned prefill plus traced decode. The qualitative log also prints the selected artifact path and runtime summary. The token-out log prints the matching selected config ID, while its exact explicit `selected_precision_config.json` invocation is recorded in `work_log.md` and `sweep_results.json`. Static policy-completeness coverage is in `tests/test_full_model.py`.

No vLLM adapter exists in this autoport stage and no vLLM integration work was started. A future adapter must construct through `build_generator`, which is the selected-policy default boundary.

## Context and non-aligned support

`context_capacity_by_kv_dtype.json` recomputes the capacity contract for both evaluated KV-cache dtypes. BFP8_B has a calculated 34,464-token physical ceiling and retains the advertised 32,768-token context. BF16 KV has an 18,304-token ceiling with the established weights, trace reservation, and runtime reserve, so it is rejected rather than reducing advertised capability. The selected CCL-only dtype change creates no persistent cache allocation and does not change the BFP8 capacity. `../context_contract.json` records the selected BF16 decode collective and unchanged 32K contract.

The one-layer policy smoke and default selected-policy check both use non-aligned prompt length 7; prefill, cache updates, and traced decode pass after the policy/layout plumbing changes.

## Post-selection token-out and qualitative checks

The normal selected-config construction path was run through the same warmed 40-layer, 128-token prompt, 128-step, no-readback token-out benchmark used by optimized-full-model. It achieved 17.879379 ms/token or 55.930356 t/s/u with exact token, position, RoPE, page-table, and first/last cache checks and no host boundary inside the timed window (`logs/bf16_ccl_token_out.log`). A matched BFP8-CCL control reached 54.456046 t/s/u (`logs/baseline_bfp8_ccl_token_out.log`), confirming the selection in the deployment-style regime while keeping it separate from Pareto ranking.

The shared-chat-template qualitative suite produced coherent, topical HF and TT completions for all six prompts, including correct French output for the translation prompt; no prompt echo, repetition loop, control-token leakage, or wrong-language behavior was found. The repeated TT run was deterministic. See `qualitative_suite/verdict.md`, `qualitative_suite/suite_summary.json`, and `logs/qualitative_suite.log`.

## Artifacts and limitations

- Tables: `sweep_results.json`, `sweep_results.csv`.
- Selected/default policy: `selected_precision_config.json`.
- Candidate policies: `candidates/*.json`.
- Capacity: `context_capacity_by_kv_dtype.json`, `../context_contract.json`.
- Plots: `top1_perf_pareto.png`, `top5_perf_pareto.png`.
- Raw evidence: `logs/*.log`.
- Qualitative evidence: `qualitative_suite/`.
- Reproduction helper: `generate_artifacts.py`.

Limitations: accuracy uses one fixed 100-token AIME24 reference rather than a broad benchmark suite; most slower candidates have one performance sample, while the baseline and winner have two; capacity is an evidence-backed allocation calculation rather than an end-to-end 32K generation run in this stage. Teardown emits the known nanobind leaked-instance diagnostic after clean device closure; no device fault or result invalidation was observed.
