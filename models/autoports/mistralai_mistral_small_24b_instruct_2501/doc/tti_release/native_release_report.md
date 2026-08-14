## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-14T17:28:40+00:00",
    "report_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_2026-08-14_17-48-38",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --runtime-model-spec-json runtime_specs/mistral_small_24b_2501_autoport_release.json --workflow release --tt-device p300x2 --tools vllm --no-auth --server-url http://127.0.0.1 --service-port 8000 --tt-metal-home /home/mvasiljevic/tt-metal --vllm-dir /home/mvasiljevic/tt-metal/vllm --limit-samples-mode ci-nightly --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_15-56-55_id_autoport_mistral-small-24b-instruct-2501_p300x2_7efNyisP.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "1529e332a1c37937a682ba04b77e7dc3418f2589",
    "vllm_commit": "6bd775d4f3a41d09d3ed03c40b45b5f9621fff9e",
    "model_impl": "mistral-small-24b-2501-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (no blocks present)
- Evals: ❌ `FAIL` (0/2 passed, 2 failed)
- Spec Tests: ✅ `PASS` (1/1 passed)

#### Blockers

- Accuracy check failed. (2 blocks)
  - `evals:LLM Eval — meta_ifeval`
  - `evals:LLM Eval — meta_gpqa_cot`
- `task:llm_benchmark`: Task 'llm_benchmark' failed (exit=1) and produced no report block.

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
| Generated      | 2026-08-14T17:28:40+00:00 |

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