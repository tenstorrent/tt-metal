## Tenstorrent Model Release Summary: Qwen/Qwen3.6-35B-A3B on P300X2

### Metadata: Qwen/Qwen3.6-35B-A3B on P300X2

```json
{
    "model_name": "Qwen/Qwen3.6-35B-A3B",
    "device": "P300X2",
    "generated_at": "2026-08-21T12:27:13+00:00",
    "report_id": "id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release_2026-08-21_13-26-34",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3.6-35B-A3B --runtime-model-spec-json /localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/tti_release/specs/qwen36_35b_a3b_autoport_release_runtime_spec.json --tt-device p300x2 --workflow release --service-port 8031 --tools vllm --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly --disable-trace-capture",
    "runtime_model_spec_json": "/localdev/vkovacevic/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/doc/tti_release/artifacts/final7/runtime_model_spec.json",
    "model_id": "id_qwen36_autoport_Qwen3.6-35B-A3B_P300X2_tti_release",
    "model_repo": "Qwen/Qwen3.6-35B-A3B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "f2bdeec2f41255c106b859601e094ad2cfa9ecca",
    "vllm_commit": "b2d90800d77ba04a54462dad1384641d17e1db47",
    "model_impl": "qwen36-autoport",
    "report_regenerated_at": "2026-08-21T13:44:41Z",
    "report_regeneration_reason": "Refresh r1_gpqa_diamond gated-dataset waiver evidence; final7 raw section data unchanged."
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (2/2 passed)
- Evals: ✅ `PASS` (1/2 passed, 1 waived)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for Qwen/Qwen3.6-35B-A3B on P300X2

| Task               | Tolerance | Published Score | Published Score Ref                                   | GPU Reference Score | gpu_reference_score_ref                               | Score | Ratio to Published | Ratio to Reference | Accuracy Check | error                         |
|:-------------------|:----------|:----------------|:------------------------------------------------------|:--------------------|:------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:------------------------------|
| leaderboard_ifeval | 0.05      | 93.09           | https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4 | 93.09               | https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4 | 89.29 | 0.9591             | 0.9591             | ✅ PASS        | N/A                           |
| r1_gpqa_diamond    | 0.1       | 86              | https://huggingface.co/Qwen/Qwen3.6-35B-A3B           | N/A                 | N/A                                                   | N/A   | N/A                | N/A                | ❌ FAIL        | no eval results parsed (rc=1) |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for Qwen/Qwen3.6-35B-A3B on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 1            | 128 | 128 | 6132.6    | 6132.6        | 6132.6        | 60.2      | 13774.0   | 9.3               | 0.073          | ✅ PASS      |

#### Target Checks

| Tier   | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| target | 1e+04       | 0.6133     | ✅ PASS    | 10.00            | 1.662           | ✅ PASS         | 7                  | 1.328             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark Targets — ISL 100 / OSL 100, concurrency 32 for Qwen/Qwen3.6-35B-A3B on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 32          | 32           | 100 | 100 | 150266.1  | 154947.0      | 154948.3      | 1003.2    | 249585.5  | 12.8              | 0.128          | ✅ PASS      |

#### Target Checks

| Tier   | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| target | 1.8e+05     | 0.8348     | ✅ PASS    | 0.80             | 1.246           | ✅ PASS         | 10                 | 1.282             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

## 📋 Summary

| Metric         | Value                     |
|:---------------|:--------------------------|
| Total Tests    | 2                         |
| Passed         | 2                         |
| Failed         | 0                         |
| Skipped        | 0                         |
| NA             | 0                         |
| Attempted      | 2                         |
| Success Rate   | 100.0%                    |
| Total Duration | 3257.11s                  |
| Total Attempts | 2                         |
| Generated      | 2026-08-21T12:27:13+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.01s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 3257.10s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for Qwen/Qwen3.6-35B-A3B on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for Qwen/Qwen3.6-35B-A3B on P300X2

| Endpoint URL                              | model_name           | Task                  |
|:------------------------------------------|:---------------------|:----------------------|
| http://127.0.0.1:8031/v1/chat/completions | Qwen/Qwen3.6-35B-A3B | vllm_chat_completions |

#### Parameter Conformance Summary

| Test Case                    | Status  | Summary    |
|:-----------------------------|:--------|:-----------|
| test_coherence_verbatim_echo | ✅ PASS | 1/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ✅ PASS | 1/1 passed |
| test_penalties               | ✅ PASS | 9/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ✅ PASS | 2/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    |
|:-----------------------------|:---------------------------------------------------------------------|:----------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |
| test_n                       | test_n[2]                                                            | ✅ PASSED |
| test_n                       | test_n[3]                                                            | ✅ PASSED |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |
| test_stop                    | test_stop[stop_seq0]                                                 | ✅ PASSED |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |
