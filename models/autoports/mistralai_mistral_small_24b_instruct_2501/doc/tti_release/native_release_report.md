## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-14T22:53:42+00:00",
    "report_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-14_22-54-25",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_spec.json --tt-device p300x2 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_cache_lookahead3/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_21-11-57_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_ojWasVTm.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "5bab286dc7fb063f4f435c840af64359fe4bf533",
    "vllm_commit": "971ee6cfcdd97a36a98e26f96ff7dda08441d219",
    "model_impl": "mistral-small-24b-2501-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/2 passed, 1 NA)
- Evals: ❌ `FAIL` (0/2 passed, 2 failed)
- Spec Tests: ❌ `FAIL` (0/1 passed, 1 failed)

#### Blockers

- Accuracy check failed. (2 blocks)
  - `evals:LLM Eval — meta_ifeval`
  - `evals:LLM Eval — meta_gpqa_cot`
- `spec.spec_tests:Vllm Chat Completions`: Vllm Chat Completions reported status=fail (attempts=1)
- `task:llm_benchmark`: Task 'llm_benchmark' failed (exit=1) after producing a report block.
- `task:spec_tests`: Task 'spec_tests' failed (exit=1) after producing a report block.

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
| 1           | 8            | 127 | 128 | 540.5     | 540.9         | 543.0         | 19.0      | 2953.8    | 43.3              | 0.339          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.386      | ✅ PASS    | 50.00            | 1.052           | ✅ PASS         | 32                 | 1.354             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects the strictest configured tier with at least one measurable check. The Target Checks table shows each configured tier — functional, complete, and/or target — from most to least lenient.

---

### vLLM Benchmark for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 127 | 0   | 2.2       | 0.0           | 0.0           | 0.0       | 73.2      | 1.5               | 389.479        |

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
| Total Duration | 0.42s                     |
| Total Attempts | 2                         |
| Generated      | 2026-08-14T22:53:42+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.01s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 0.41s    | 1        | vLLM chat/completions parameter conformance       |

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
| test_coherence_verbatim_echo | ❌ FAIL | 0/1 passed |
| test_determinism_parameters  | ❌ FAIL | 0/3 passed |
| test_logprobs                | ❌ FAIL | 0/1 passed |
| test_max_tokens              | ❌ FAIL | 0/2 passed |
| test_n                       | ❌ FAIL | 0/2 passed |
| test_non_uniform_seeding     | ❌ FAIL | 0/1 passed |
| test_penalties               | ❌ FAIL | 0/9 passed |
| test_seed_reproducibility    | ❌ FAIL | 0/1 passed |
| test_stop                    | ❌ FAIL | 0/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    | Message                                                                                                                                                                                                                                                       |
|:-----------------------------|:---------------------------------------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_logprobs                | test_logprobs                                                        | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_max_tokens              | test_max_tokens[10]                                                  | ❌ FAILED | Traceback: KeyError: 'usage'                                                                                                                                                                                                                                  |
| test_max_tokens              | test_max_tokens[5]                                                   | ❌ FAILED | Traceback: KeyError: 'usage'                                                                                                                                                                                                                                  |
| test_n                       | test_n[2]                                                            | ❌ FAILED | Traceback: AssertionError: AssertionError: choices field is not in response assert 'choices' in {'error': {'code': 500, 'message': 'EngineCore encountered an issue. See stack trace (above) for the root cause.', 'param': None, 'type': 'InternalServerE... |
| test_n                       | test_n[3]                                                            | ❌ FAILED | Traceback: AssertionError: AssertionError: choices field is not in response assert 'choices' in {'error': {'code': 500, 'message': 'EngineCore encountered an issue. See stack trace (above) for the root cause.', 'param': None, 'type': 'InternalServerE... |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ❌ FAILED | Traceback: Failed: Request failed for seed 0: 'choices'                                                                                                                                                                                                       |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_stop                    | test_stop[stop_seq0]                                                 | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
| test_stop                    | test_stop[stop_seq1]                                                 | ❌ FAILED | Traceback: KeyError: 'choices'                                                                                                                                                                                                                                |
