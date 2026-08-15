## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-15T02:11:57.830327+00:00",
    "report_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_final_reports_merge_2026-08-15_02-10-04",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "native reports-only schema merge: retained release eval blocks + corrected benchmarks report + corrected spec_tests report",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_spec.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "5bab286dc7fb063f4f435c840af64359fe4bf533",
    "vllm_commit": "aab6d846caf95c5e9cf8038f3338650a9132c383",
    "model_impl": "mistral-small-24b-2501-autoport",
    "reports_only_aggregation": true,
    "source_reports": [
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_cache_lookahead3/workflow_logs/reports_output/release/data/report_data_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-14_22-54-25.json",
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/component_cache_slotfix/workflow_logs/reports_output/benchmarks/data/report_data_mistralai__Mistral-Small-24B-Instruct-2501_2026-08-15T015146+0000.json",
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/spec_cache_slotfix/workflow_logs/reports_output/spec_tests/data/report_data_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-15_02-10-04.json"
    ],
    "source_report_sha256": {
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_cache_lookahead3/workflow_logs/reports_output/release/data/report_data_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-14_22-54-25.json": "df4adc372f6280208219828cca975a3f15d45a982b74d262dd47a41bc9edfc9c",
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/component_cache_slotfix/workflow_logs/reports_output/benchmarks/data/report_data_mistralai__Mistral-Small-24B-Instruct-2501_2026-08-15T015146+0000.json": "ca6d2a0d237aff00b4e0337789b4270fea035336a5b0819ad18a8724137fc9a1",
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/spec_cache_slotfix/workflow_logs/reports_output/spec_tests/data/report_data_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-15_02-10-04.json": "fd1dfe160ae3d2d7f0e7772f3dd149e9183bd6df888e5f1c955bfe9a904f9a27"
    }
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/13 passed, 12 NA)
- Evals: ❌ `FAIL` (0/2 passed, 2 failed)
- Spec Tests: ✅ `PASS` (1/1 passed)

#### Blockers

- Accuracy check failed. (2 blocks)
  - `evals:LLM Eval — meta_ifeval`
  - `evals:LLM Eval — meta_gpqa_cot`

---

### Accuracy Evaluations for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Task          | executed_task_name            | Tolerance | Published Score | Published Score Ref                                                                            | GPU Reference Score | gpu_reference_score_ref                                                                        | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:--------------|:------------------------------|:----------|:----------------|:-----------------------------------------------------------------------------------------------|:--------------------|:-----------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|
| meta_ifeval   | ifeval                        | 0.05      | 82.9            | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 82.9                | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 72.56 | 0.8752             | 0.8752             | ❌ FAIL        |
| meta_gpqa_cot | gpqa_main_official_cot_n_shot | 0.05      | 45.3            | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 45.3                | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 38.89 | 0.8585             | 0.8585             | ❌ FAIL        |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 127 | 128 | 535.4     | 535.4         | 536.3         | 18.9      | 2940.2    | 43.5              | 0.340          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.3824     | ✅ PASS    | 50.00            | 1.056           | ✅ PASS         | 32                 | 1.36              | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects the strictest configured tier with at least one measurable check. The Target Checks table shows each configured tier — functional, complete, and/or target — from most to least lenient.

---

### vLLM Benchmark for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 127   | 128  | 1231.9    | 1173.2        | 1927.2        | 19.5      | 3704.2    | 1105.5            | 8.637          |
| 1           | 4            | 127   | 1024 | 539.1     | 536.4         | 548.7         | 18.4      | 19409.2   | 52.8              | 0.051          |
| 28          | 112          | 127   | 1024 | 1318.5    | 1166.0        | 1902.7        | 18.9      | 20612.0   | 1391.0            | 1.358          |
| 1           | 4            | 1023  | 128  | 25238.0   | 25244.3       | 25291.6       | 20.0      | 27784.1   | 4.6               | 0.036          |
| 28          | 112          | 1023  | 128  | 49765.1   | 50517.2       | 51133.6       | 27.8      | 53293.5   | 67.2              | 0.525          |
| 1           | 4            | 2047  | 128  | 77666.3   | 77630.1       | 77906.8       | 21.4      | 80387.3   | 1.6               | 0.012          |
| 15          | 60           | 2047  | 128  | 150079.8  | 155188.9      | 155342.0      | 62.2      | 157978.5  | 12.2              | 0.095          |
| 1           | 4            | 4095  | 128  | 182447.9  | 182430.3      | 182738.3      | 24.0      | 185500.6  | 0.7               | 0.005          |
| 7           | 28           | 4095  | 128  | 339133.5  | 365065.6      | 365758.6      | 229.6     | 368295.1  | 2.4               | 0.019          |
| 1           | 2            | 8191  | 128  | 393025.0  | 393025.0      | 393085.3      | 29.2      | 396737.5  | 0.3               | 0.003          |
| 3           | 6            | 8191  | 128  | 524311.3  | 393240.6      | 786999.7      | 545.2     | 593555.0  | 0.6               | 0.005          |
| 1           | 2            | 16383 | 128  | 814523.9  | 814523.9      | 814762.0      | 39.7      | 819560.8  | 0.2               | 0.001          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

---

## 📋 Summary

| Metric         | Value                            |
|:---------------|:---------------------------------|
| Total Tests    | 2                                |
| Passed         | 2                                |
| Failed         | 0                                |
| Skipped        | 0                                |
| NA             | 0                                |
| Attempted      | 2                                |
| Success Rate   | 100.0%                           |
| Total Duration | 1084.81s                         |
| Total Attempts | 2                                |
| Generated      | 2026-08-15T02:11:57.830327+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1084.81s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Endpoint URL                              | model_name                                | Task                  |
|:------------------------------------------|:------------------------------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | mistralai/Mistral-Small-24B-Instruct-2501 | vllm_chat_completions |

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
