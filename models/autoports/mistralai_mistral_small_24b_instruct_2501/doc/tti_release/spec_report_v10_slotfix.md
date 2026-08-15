## Tenstorrent Model Release Summary: Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-15 02:10:04",
    "report_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_2026-08-15_02-10-04",
    "workflow": "spec_tests",
    "server_mode": "API",
    "run_command": "python run.py --workflow spec_tests --runtime-model-spec-json /home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_spec.json --tt-device p300x2 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/spec_cache_slotfix/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-15_01-51-59_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_gVATv6fP.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "5bab286dc7fb063f4f435c840af64359fe4bf533",
    "vllm_commit": "aab6d846caf95c5e9cf8038f3338650a9132c383",
    "model_impl": "mistral-small-24b-2501-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (no blocks present)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

## 📋 Summary

| Metric         | Value               |
|:---------------|:--------------------|
| Total Tests    | 2                   |
| Passed         | 2                   |
| Failed         | 0                   |
| Skipped        | 0                   |
| NA             | 0                   |
| Attempted      | 2                   |
| Success Rate   | 100.0%              |
| Total Duration | 1084.81s            |
| Total Attempts | 2                   |
| Generated      | 2026-08-15 02:10:04 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1084.81s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for Mistral-Small-24B-Instruct-2501 on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for Mistral-Small-24B-Instruct-2501 on P300X2

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
