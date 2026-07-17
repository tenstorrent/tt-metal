## Tenstorrent Model Release Summary: google/gemma-4-31B on P150X4

### Metadata: google/gemma-4-31B on P150X4

```json
{
    "model_name": "google/gemma-4-31B",
    "device": "P150X4",
    "generated_at": "2026-07-16T21:52:26+00:00",
    "report_id": "id_autoport_google_gemma_4_31b_p150x4_release_2026-07-16_23-53-05",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model gemma-4-31B-it --runtime-model-spec-json ../autoport_release_spec.json --tt-device p150x4 --workflow release --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/localdev/odjuricic/tt-metal/.exp_run/tti-release/gemma4-31b-20260716/release_cache_final6/workflow_logs/runtime_model_specs/runtime_model_spec_2026-07-16_20-46-55_id_autoport_google_gemma_4_31b_p150x4_release_XoOYIVRt.json",
    "model_id": "id_autoport_google_gemma_4_31b_p150x4_release",
    "model_repo": "google/gemma-4-31B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "2be0f245e20",
    "vllm_commit": "44b7853",
    "model_impl": "autoport-google-gemma-4-31b"
}
```

### Acceptance Criteria

- Acceptance status: `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: `PASS` (1/1 passed)
- Evals: `PASS` (0/2 passed, 2 waived)
- Spec Tests: `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for google/gemma-4-31B on P150X4

|     Task      | Tolerance |            Published Score Ref            | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|---------------|-----------|-------------------------------------------|-------|--------------------|--------------------|----------------|
| meta_ifeval   |      0.05 | https://huggingface.co/google/gemma-4-31B | 25.18 | N/A                | N/A                | NA             |
| meta_gpqa_cot |      0.05 | https://huggingface.co/google/gemma-4-31B | 20.98 | N/A                | N/A                | NA             |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vllm Benchmark Targets for google/gemma-4-31B on P150X4

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|-------------|--------------|-----|-----|-----------|---------------|---------------|-----------|-----------|-------------------|----------------|
|           1 |            8 | 127 | 128 |     197.3 |         191.2 |         228.5 |      38.2 |    5043.3 |              25.4 |          0.198 |

#### Target Checks

|        Tier         | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Check |
|---------------------|-------------|------------|------------|------------------|-----------------|-----------------|-------------------|
| customer_functional |        1000 |     0.1973 | PASS       |            20.00 |           1.269 | PASS            | NA                |
| customer_complete   |         700 |     0.2818 | PASS       |            24.00 |           1.057 | PASS            | NA                |
| customer_sellable   |         600 |     0.3288 | PASS       |            25.00 |           1.015 | PASS            | NA                |

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### Vllm for google/gemma-4-31B on P150X4

| Concurrency | Num Requests |  ISL  | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|-------------|--------------|-------|------|-----------|---------------|---------------|-----------|-----------|-------------------|----------------|
|          32 |          256 |   127 |  128 |    4950.9 |        5109.4 |        5390.2 |      72.8 |   14194.9 |             288.5 |          2.254 |
|           1 |            4 |   127 | 1024 |     205.8 |         207.5 |         220.3 |      31.1 |   32006.6 |              32.0 |          0.031 |
|          32 |          128 |   127 | 1024 |    4944.5 |        4908.3 |        5366.3 |      62.8 |   69236.3 |             473.3 |          0.462 |
|           1 |            4 |  1023 |  128 |     324.0 |         318.4 |         342.2 |      39.2 |    5308.5 |              24.1 |          0.188 |
|          32 |          128 |  1023 |  128 |    8652.7 |        8879.6 |        8978.9 |      81.8 |   19037.6 |             215.1 |          1.681 |
|           1 |            4 |  2047 |  128 |     527.3 |         520.1 |         549.5 |      39.4 |    5528.2 |              23.2 |          0.181 |
|          32 |          128 |  2047 |  128 |   14830.8 |       15329.1 |       15487.8 |      84.8 |   25603.3 |             160.0 |          1.250 |
|           1 |            4 |  4095 |  128 |     953.8 |         952.2 |         959.1 |      39.5 |    5975.4 |              21.4 |          0.167 |
|          26 |          104 |  4095 |  128 |   22905.6 |       23838.9 |       23849.3 |      84.9 |   33688.6 |              98.8 |          0.772 |
|           1 |            2 |  8191 |  128 |    1989.3 |        1989.3 |        1995.0 |      39.6 |    7019.6 |              18.2 |          0.142 |
|          13 |           26 |  8191 |  128 |   23450.9 |       25212.9 |       25223.8 |      78.5 |   33420.0 |              49.8 |          0.389 |
|           1 |            2 | 16383 |  128 |    4239.8 |        4239.8 |        4240.5 |      39.7 |    9282.5 |              13.8 |          0.108 |
|           6 |           12 | 16383 |  128 |   21742.1 |       25186.9 |       25204.6 |      83.7 |   32373.1 |              23.7 |          0.185 |
|           1 |            1 | 32767 |  128 |   10602.4 |       10602.4 |       10602.4 |      40.8 |   15782.9 |               8.1 |          0.063 |
|           3 |            3 | 32767 |  128 |   24841.5 |       31842.3 |       31844.2 |     109.0 |   38684.0 |               9.9 |          0.077 |
|           1 |            1 | 65535 |  128 |   26812.3 |       26812.3 |       26812.3 |      41.7 |   32106.3 |               4.0 |          0.031 |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

---

## 📋 Summary

| Metric | Value |
|:-------|------:|
| Total Tests | 2 |
| Passed | 2 |
| Failed | 0 |
| Skipped | 0 |
| NA | 0 |
| Attempted | 2 |
| Success Rate | 100.0% |
| Total Duration | 5807.43s |
| Total Attempts | 2 |
| Generated | 2026-07-16T21:52:26+00:00 |

## 🧪 Test Results

| Status | Test Name | Duration | Attempts | Description |
|:------:|:----------|---------:|---------:|:------------|
| ✅ | LoggerForkSafetyTest | 0.01s | 1 | Test for logging fork safety to prevent deadlocks |
| ✅ | VLLMParamConformanceTest | 5807.42s | 1 | vLLM chat/completions parameter conformance |

---

### Logger Fork Safety for google/gemma-4-31B on P150X4

| Success | Child Result | Status | Attempts |
|---------|--------------|--------|----------|
| true    | OK           | pass   |        1 |

---

### Vllm Chat Completions for google/gemma-4-31B on P150X4

|               Endpoint URL                |     model_name     |         Task          | Success | Status | Attempts |
|-------------------------------------------|--------------------|-----------------------|---------|--------|----------|
| http://127.0.0.1:8000/v1/chat/completions | google/gemma-4-31B | vllm_chat_completions | true    | pass   |        1 |

#### Parameter Conformance Summary

|          Test Case          | Status  |  Summary   |
|-----------------------------|---------|------------|
| test_determinism_parameters | ✅ PASS | 3/3 passed |
| test_logprobs               | ✅ PASS | 1/1 passed |
| test_max_tokens             | ✅ PASS | 2/2 passed |
| test_n                      | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding    | ✅ PASS | 1/1 passed |
| test_penalties              | ✅ PASS | 9/9 passed |
| test_seed_reproducibility   | ✅ PASS | 1/1 passed |
| test_stop                   | ✅ PASS | 2/2 passed |

#### Detailed Test Results

|          Test Case          |                           Parametrization                            |  Status   |
|-----------------------------|----------------------------------------------------------------------|-----------|
| test_determinism_parameters | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |
| test_determinism_parameters | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |
| test_determinism_parameters | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |
| test_logprobs               | test_logprobs                                                        | ✅ PASSED |
| test_max_tokens             | test_max_tokens[10]                                                  | ✅ PASSED |
| test_max_tokens             | test_max_tokens[5]                                                   | ✅ PASSED |
| test_n                      | test_n[2]                                                            | ✅ PASSED |
| test_n                      | test_n[3]                                                            | ✅ PASSED |
| test_non_uniform_seeding    | test_non_uniform_seeding                                             | ✅ PASSED |
| test_penalties              | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |
| test_penalties              | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |
| test_penalties              | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |
| test_penalties              | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |
| test_penalties              | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |
| test_penalties              | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |
| test_penalties              | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |
| test_penalties              | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |
| test_penalties              | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |
| test_seed_reproducibility   | test_seed_reproducibility                                            | ✅ PASSED |
| test_stop                   | test_stop[stop_seq0]                                                 | ✅ PASSED |
| test_stop                   | test_stop[stop_seq1]                                                 | ✅ PASSED |
