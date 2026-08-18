## Tenstorrent Model Release Summary: Qwen3.6-27B on P300X2

### Metadata: Qwen3.6-27B on P300X2

```json
{
    "model_name": "Qwen3.6-27B",
    "device": "P300X2",
    "generated_at": "2026-08-18 15:37:14",
    "report_id": "id_qwen36-blackhole_Qwen3.6-27B_p300x2_2026-08-18_15-37-14",
    "workflow": "spec_tests",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3.6-27B --workflow spec_tests --tt-device p300x2 --local-server --service-port 8000 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly --ci-mode --tt-metal-home /home/mvasiljevic/tt-metal --tt-metal-python-venv-dir /home/mvasiljevic/tt-metal/python_env --vllm-dir /home/mvasiljevic/vllm --override-tt-config '{\"trace_region_size\": 200000000}'",
    "runtime_model_spec_json": "/home/mvasiljevic/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-18_15-37-09_id_qwen36-blackhole_Qwen3.6-27B_p300x2_RAtf6Kf5.json",
    "model_id": "id_qwen36-blackhole_Qwen3.6-27B_p300x2",
    "model_repo": "Qwen/Qwen3.6-27B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "de59f8a",
    "vllm_commit": "03fa3af",
    "model_impl": "qwen36-blackhole"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (no blocks present)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: ❌ `FAIL` (0/1 passed, 1 failed)

#### Blockers

- `spec.spec_tests:Vllm Chat Completions`: Vllm Chat Completions reported status=fail (attempts=1)
- `task:spec_tests`: Task 'spec_tests' failed (exit=1) after producing a report block.

---

## 📋 Summary

| Metric         | Value               |
|:---------------|:--------------------|
| Total Tests    | 2                   |
| Passed         | 1                   |
| Failed         | 1                   |
| Skipped        | 0                   |
| NA             | 0                   |
| Attempted      | 2                   |
| Success Rate   | 50.0%               |
| Total Duration | 1.44s               |
| Total Attempts | 2                   |
| Generated      | 2026-08-18 15:37:14 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 1.43s    | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for Qwen3.6-27B on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for Qwen3.6-27B on P300X2

| Endpoint URL                              | model_name       | Task                  |
|:------------------------------------------|:-----------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | Qwen/Qwen3.6-27B | vllm_chat_completions |

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
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_logprobs                | test_logprobs                                                        | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_max_tokens              | test_max_tokens[10]                                                  | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_max_tokens              | test_max_tokens[5]                                                   | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_n                       | test_n[2]                                                            | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_n                       | test_n[3]                                                            | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ❌ FAILED | Traceback: Failed: Request failed for seed 0: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new c... |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_stop                    | test_stop[stop_seq0]                                                 | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |
| test_stop                    | test_stop[stop_seq1]                                                 | ❌ FAILED | Traceback: requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /v1/chat/completions (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=8000): Failed to establish a new... |