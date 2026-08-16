## Tenstorrent Model Release Summary: meta-models/Muse-Glimmer-30B on P300X2

### Metadata: meta-models/Muse-Glimmer-30B on P300X2

```json
{
    "model_name": "meta-models/Muse-Glimmer-30B",
    "device": "P300X2",
    "generated_at": "2026-08-16T07:35:08+00:00",
    "report_id": "id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_2026-08-16_04-04-24",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Muse-Glimmer-30B --runtime-model-spec-json /home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/specs/muse_glimmer_30b_autoport_release.json --tt-device p300x2 --workflow release --service-port 8000 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/cache_root/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-16_02-39-22_id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_eqbjPKm1.json",
    "model_id": "id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2",
    "model_repo": "meta-models/Muse-Glimmer-30B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "7db0eca",
    "vllm_commit": null,
    "model_impl": "muse-glimmer-30b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/18 passed, 17 NA)
- Evals: ✅ `PASS` (2/2 passed)
- Spec Tests: ❌ `FAIL` (0/1 passed, 1 failed)

#### Blockers

- `spec.spec_tests:Vllm Chat Completions`: Vllm Chat Completions reported status=fail (attempts=1)
- `task:spec_tests`: Task 'spec_tests' failed (exit=1) after producing a report block.

---

### Accuracy Evaluations for meta-models/Muse-Glimmer-30B on P300X2

| Task   | Tolerance | Published Score | Published Score Ref                                                                              | GPU Reference Score | gpu_reference_score_ref                                                                                                                   | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:-------|:----------|:----------------|:-------------------------------------------------------------------------------------------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|
| ifeval | 0.05      | 77              | https://huggingface.co/meta-models/Muse-Glimmer-30B (model card, IFBench 77.0)                   | 77                  | vendor-published IFBench score used as a conservative floor for the IFEval row; no Tenstorrent GPU control run exists for this checkpoint | 94.45 | 1.227              | 1.227              | ✅ PASS        |
| aime25 | 0.1       | 94.7            | https://huggingface.co/meta-models/Muse-Glimmer-30B (model card, AIME 2026 94.7, High Reasoning) | 94.7                | vendor-published AIME score used as the reference; no Tenstorrent GPU control run exists for this checkpoint                              | 90    | 0.9504             | 0.9504             | ✅ PASS        |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for meta-models/Muse-Glimmer-30B on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 127 | 128 | 72.1      | 71.7          | 74.9          | 23.0      | 2996.3    | 42.7              | 0.334          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 88.3        | 0.8167     | ✅ PASS    | 11.33            | 3.833           | ✅ PASS         | 11.33              | 3.77              | ✅ PASS           |
| complete   | 17.66       | 4.084      | ❌ FAIL    | 56.65            | 0.7666          | ❌ FAIL         | 56.65              | 0.754             | ❌ FAIL           |
| target     | 8.83        | 8.167      | ❌ FAIL    | 113.30           | 0.3833          | ❌ FAIL         | 113.3              | 0.377             | ❌ FAIL           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for meta-models/Muse-Glimmer-30B on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 127   | 128  | 2052.3    | 2045.8        | 2114.2        | 23.1      | 4985.6    | 821.4             | 6.417          |
| 1           | 4            | 127   | 1024 | 71.4      | 71.2          | 73.5          | 23.5      | 24090.9   | 42.5              | 0.042          |
| 32          | 128          | 127   | 1024 | 2050.1    | 2046.1        | 2067.1        | 24.0      | 26638.5   | 1230.0            | 1.201          |
| 1           | 4            | 1023  | 128  | 154.5     | 153.9         | 156.8         | 23.9      | 3187.9    | 40.1              | 0.314          |
| 32          | 128          | 1023  | 128  | 4687.3    | 4686.2        | 4719.4        | 24.7      | 7822.2    | 523.5             | 4.090          |
| 1           | 4            | 2047  | 128  | 270.7     | 271.6         | 273.6         | 24.2      | 3343.1    | 38.3              | 0.299          |
| 32          | 128          | 2047  | 128  | 8181.4    | 8177.6        | 8200.1        | 25.7      | 11442.2   | 357.9             | 2.796          |
| 1           | 4            | 4095  | 128  | 490.2     | 489.1         | 495.2         | 24.5      | 3596.6    | 35.6              | 0.278          |
| 32          | 128          | 4095  | 128  | 15167.6   | 15138.8       | 15214.3       | 26.2      | 18490.1   | 221.5             | 1.730          |
| 1           | 2            | 8191  | 128  | 987.0     | 987.0         | 990.4         | 24.9      | 4150.1    | 30.8              | 0.241          |
| 32          | 64           | 8191  | 128  | 30355.8   | 30813.6       | 30927.6       | 30.4      | 34212.2   | 119.7             | 0.935          |
| 1           | 2            | 16383 | 128  | 2187.6    | 2187.6        | 2191.2        | 25.9      | 5472.5    | 23.4              | 0.183          |
| 32          | 64           | 16383 | 128  | 55698.9   | 54101.2       | 69299.8       | 135.4     | 72892.9   | 56.2              | 0.439          |
| 1           | 1            | 32767 | 128  | 4769.7    | 4769.7        | 4769.7        | 27.6      | 8278.8    | 15.5              | 0.121          |
| 31          | 31           | 32767 | 128  | 98419.1   | 98312.7       | 147453.4      | 420.4     | 151803.9  | 26.1              | 0.204          |
| 1           | 1            | 65535 | 128  | 10836.0   | 10836.0       | 10836.0       | 31.5      | 14837.6   | 8.6               | 0.067          |
| 16          | 16           | 65535 | 128  | 118210.3  | 121337.6      | 179302.7      | 521.6     | 184452.3  | 10.4              | 0.081          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

---

## 📋 Summary

| Metric         | Value                     |
|:---------------|:--------------------------|
| Total Tests    | 2                         |
| Passed         | 1                         |
| Failed         | 1                         |
| Skipped        | 0                         |
| NA             | 0                         |
| Attempted      | 2                         |
| Success Rate   | 50.0%                     |
| Total Duration | 368.19s                   |
| Total Attempts | 2                         |
| Generated      | 2026-08-16T07:35:08+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 368.18s  | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for meta-models/Muse-Glimmer-30B on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for meta-models/Muse-Glimmer-30B on P300X2

| Endpoint URL                              | model_name                   | Task                  |
|:------------------------------------------|:-----------------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | meta-models/Muse-Glimmer-30B | vllm_chat_completions |

#### Parameter Conformance Summary

| Test Case                    | Status  | Summary    |
|:-----------------------------|:--------|:-----------|
| test_coherence_verbatim_echo | ✅ PASS | 1/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ✅ PASS | 1/1 passed |
| test_penalties               | ❌ FAIL | 8/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ✅ PASS | 2/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    | Message                                                                                                                                                                                                                                                       |
|:-----------------------------|:---------------------------------------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_n                       | test_n[2]                                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_n                       | test_n[3]                                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ❌ FAILED | Traceback: AssertionError: Test failed: Penalty unexpectedly reduced diversity. assert 0.12376237623762376 >= (0.15079365079365079 * 0.9). Base: The man woke up. The man woke up. The man woke up.  He got out of bed. He got out of bed. He got out of b... |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_stop                    | test_stop[stop_seq0]                                                 | ✅ PASSED |                                                                                                                                                                                                                                                               |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |                                                                                                                                                                                                                                                               |
