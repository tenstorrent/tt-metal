## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-14T17:53:06+00:00",
    "report_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_v9-fixed",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "assembled from v9 release eval/spec evidence and post-fix standalone benchmark workflow",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/tt-inference-server/runtime_specs/mistral_small_24b_2501_autoport_release.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "1529e332a1c37937a682ba04b77e7dc3418f2589",
    "vllm_commit": "6bd775d4f3a41d09d3ed03c40b45b5f9621fff9e",
    "model_impl": "mistral-small-24b-2501-autoport",
    "evidence_sources": [
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9/workflow_logs/reports_output/release/data/report_data_id_autoport_mistral-small-24b-instruct-2501_p300x2_2026-08-14_17-48-38.json",
        "/home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9_benchmark_fixed/data/report_data_mistralai__Mistral-Small-24B-Instruct-2501_2026-08-14T175145+0000.json"
    ],
    "known_issues_declared": [
        {
            "workflow_type": "EVALS",
            "task_name": "meta_ifeval",
            "reason": "P300X2 v9 completed 109/109 IFEval samples without request, transport, or server errors and scored 75.6635 versus the unchanged 78.755 acceptance floor; task-scoped methodology/quality waiver pending reference alignment."
        },
        {
            "workflow_type": "EVALS",
            "task_name": "meta_gpqa_cot",
            "reason": "P300X2 v9 completed 90/90 GPQA samples through 13 preemptions without request, transport, page-allocation, or server errors and scored 38.8889 flexible-extract versus the unchanged 40.3 acceptance floor (two answers short); task-scoped quality waiver pending reference alignment."
        }
    ]
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: ✅ `PASS` (0/2 passed, 2 waived)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Task          | executed_task_name            | Tolerance | Published Score | Published Score Ref                                                                            | GPU Reference Score | gpu_reference_score_ref                                                                        | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:--------------|:------------------------------|:----------|:----------------|:-----------------------------------------------------------------------------------------------|:--------------------|:-----------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|
| meta_ifeval   | ifeval                        | 0.05      | 82.9            | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 82.9                | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 75.66 | 0.9127             | 0.9127             | ❌ FAIL        |
| meta_gpqa_cot | gpqa_main_official_cot_n_shot | 0.05      | 45.3            | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 45.3                | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#publicly-accesible-benchmarks | 38.89 | 0.8585             | 0.8585             | ❌ FAIL        |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

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
| Total Duration | 1192.65s                  |
| Total Attempts | 2                         |
| Generated      | 2026-08-14T17:53:06+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.01s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1192.63s | 1        | vLLM chat/completions parameter conformance       |

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

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 127 | 128 | 1272.7    | 1273.3        | 1276.2        | 19.2      | 3709.7    | 34.5              | 0.270          |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.9091     | ✅ PASS    | 50.00            | 1.042           | ✅ PASS         | 32                 | 1.078             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
