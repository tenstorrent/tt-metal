## Tenstorrent Model Release Summary: zai-org/GLM-4.7-Flash on P150

### Metadata: zai-org/GLM-4.7-Flash on P150

```json
{
    "model_name": "zai-org/GLM-4.7-Flash",
    "device": "P150",
    "generated_at": "2026-09-04T20:55:46+00:00",
    "report_id": "id_autoport-glm47-flash_GLM-4.7-Flash_p150_2026-09-04_20-48-56",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model GLM-4.7-Flash --dev-mode --tt-device p150 --workflow release --service-port 8000 --limit-samples-mode ci-nightly --no-auth --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/stisi/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-04_15-40-09_id_autoport-glm47-flash_GLM-4.7-Flash_p150_xUUhcIYO.json",
    "model_id": "id_autoport-glm47-flash_GLM-4.7-Flash_p150",
    "model_repo": "zai-org/GLM-4.7-Flash",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "autoport-glm47-flash"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (1/23 passed, 22 NA)
- Evals: ✅ `PASS` (0/2 passed, 1 waived, 1 NA)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for zai-org/GLM-4.7-Flash on P150

| Task                      | Tolerance | Published Score | Published Score Ref                                                     | Score | Ratio to Published | Ratio to Reference | Accuracy Check | mean_seconds_per_task |
|:--------------------------|:----------|:----------------|:------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:----------------------|
| ifeval                    | 0.05      | N/A             | N/A                                                                     | 71.43 | N/A                | N/A                | 🟨 NA          | 11.95                 |
| gpqa_diamond_cot_zeroshot | 0.05      | 75.2            | https://huggingface.co/zai-org/GLM-4.7-Flash#performances-on-benchmarks | 70    | 0.9309             | N/A                | ❌ FAIL        | 420.1                 |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for zai-org/GLM-4.7-Flash on P150

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 296.1     | 300.4         | 310.2         | 29.5      | 4042.0    | 31.7             | 31.7              | 63.3             | 0.247          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Output Target | Tput Output Ratio | Tput Output Check | TTFT Target | TTFT Ratio |
|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|:------------|:-----------|
| functional | 🟨 NA      | 25.64            | 1.322           | ✅ PASS         | 25.64              | 1.235             | ✅ PASS           | N/A         | N/A        |
| complete   | 🟨 NA      | 128.20           | 0.2645          | ❌ FAIL         | 128.2              | 0.247             | ❌ FAIL           | N/A         | N/A        |
| target     | ❌ FAIL    | 33.90            | 1               | ✅ PASS         | 31.84              | 0.9946            | ✅ PASS           | 274.1       | 1.08       |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for zai-org/GLM-4.7-Flash on P150

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 32          | 256          | 128    | 128  | 9317.9    | 9324.8        | 9393.6        | 91.6      | 20950.7   | 195.5            | 195.5             | 391.0            | 1.527          |
| 1           | 4            | 128    | 1024 | 290.3     | 289.6         | 303.7         | 30.7      | 31722.3   | 4.0              | 32.3              | 36.3             | 0.032          |
| 32          | 128          | 128    | 1024 | 9347.9    | 9358.8        | 9391.3        | 99.0      | 110627.9  | 37.0             | 296.2             | 333.2            | 0.289          |
| 1           | 4            | 1024   | 128  | 1961.1    | 1961.3        | 2045.5        | 31.4      | 5953.2    | 172.0            | 21.5              | 193.5            | 0.168          |
| 32          | 128          | 1024   | 128  | 64251.3   | 64265.3       | 64424.3       | 93.6      | 76137.2   | 430.4            | 53.8              | 484.2            | 0.420          |
| 1           | 4            | 2048   | 128  | 4042.5    | 4062.5        | 4171.0        | 33.6      | 8311.6    | 246.4            | 15.4              | 261.8            | 0.120          |
| 32          | 128          | 2048   | 128  | 131504.4  | 131529.6      | 131977.1      | 96.4      | 143740.9  | 455.9            | 28.5              | 484.4            | 0.223          |
| 1           | 4            | 4096   | 128  | 8492.3    | 8553.1        | 8700.8        | 37.2      | 13222.9   | 309.8            | 9.7               | 319.4            | 0.076          |
| 32          | 128          | 4096   | 128  | 275014.2  | 275402.0      | 275969.7      | 99.8      | 287683.7  | 455.6            | 14.2              | 469.8            | 0.111          |
| 1           | 2            | 8192   | 128  | 18122.7   | 18122.7       | 18392.2       | 45.2      | 23865.1   | 343.3            | 5.4               | 348.6            | 0.042          |
| 24          | 48           | 8192   | 128  | 449201.4  | 449174.5      | 450633.8      | 103.4     | 462332.4  | 425.3            | 6.6               | 431.9            | 0.052          |
| 1           | 2            | 8192   | 1024 | 18128.8   | 18128.8       | 18395.3       | 46.1      | 65244.1   | 125.6            | 15.7              | 141.3            | 0.015          |
| 22          | 44           | 8192   | 1024 | 411401.6  | 411376.4      | 412720.9      | 108.1     | 521984.3  | 345.3            | 43.2              | 388.4            | 0.042          |
| 1           | 2            | 10000  | 1024 | 23478.9   | 23478.9       | 23839.3       | 49.5      | 74137.6   | 134.9            | 13.8              | 148.7            | 0.013          |
| 18          | 36           | 10000  | 1024 | 434613.3  | 434589.6      | 435880.1      | 106.2     | 543245.9  | 331.3            | 33.9              | 365.3            | 0.033          |
| 1           | 2            | 16384  | 128  | 42454.5   | 42454.5       | 42963.8       | 60.4      | 50129.5   | 326.8            | 2.6               | 329.4            | 0.020          |
| 12          | 24           | 16384  | 128  | 522536.0  | 522512.0      | 524101.9      | 105.4     | 535923.1  | 366.9            | 2.9               | 369.7            | 0.022          |
| 1           | 1            | 32768  | 128  | 108869.0  | 108869.0      | 108869.0      | 91.6      | 120505.9  | 271.9            | 1.1               | 273.0            | 0.008          |
| 6           | 6            | 32768  | 128  | 668893.9  | 668903.0      | 668905.1      | 111.6     | 683070.4  | 287.8            | 1.1               | 289.0            | 0.009          |
| 1           | 1            | 65536  | 128  | 319610.1  | 319610.1      | 319610.1      | 153.2     | 339066.0  | 193.3            | 0.4               | 193.7            | 0.003          |
| 3           | 3            | 65536  | 128  | 953219.1  | 953234.4      | 953235.7      | 164.7     | 974130.5  | 201.8            | 0.4               | 202.2            | 0.003          |
| 1           | 1            | 131072 | 128  | 1034101.9 | 1034101.9     | 1034101.9     | 277.1     | 1069299.0 | 122.6            | 0.1               | 122.7            | 0.001          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

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
| Total Duration | 621.20s                   |
| Total Attempts | 2                         |
| Generated      | 2026-09-04T20:55:46+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                                                                     |
|:--------|:-------------------------|:---------|:---------|:------------------------------------------------------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks                                               |
| ✅ PASS | VLLMParamConformanceTest | 621.19s  | 1        | vLLM chat/completions parameter conformance (GLM-4.7-Flash autoport, non-thinking request mode) |

---

### Logger Fork Safety for zai-org/GLM-4.7-Flash on P150

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for zai-org/GLM-4.7-Flash on P150

| Endpoint URL                              | model_name            | Task                  |
|:------------------------------------------|:----------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | zai-org/GLM-4.7-Flash | vllm_chat_completions |

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
