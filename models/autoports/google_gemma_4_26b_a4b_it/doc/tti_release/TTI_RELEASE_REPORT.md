## Tenstorrent Model Release Summary: google/gemma-4-26B-A4B-it on P300X2

### Metadata: google/gemma-4-26B-A4B-it on P300X2

```json
{
    "model_name": "google/gemma-4-26B-A4B-it",
    "device": "P300X2",
    "generated_at": "2026-08-16T09:38:13+00:00",
    "report_id": "id_autoport-Gemma-4-26B-A4B-it_p300x2_release_2026-08-16_09-56-05",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/release_gate_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-16_06-11-48_id_autoport-Gemma-4-26B-A4B-it_p300x2_release_4w0z1WvB.json",
    "model_id": "id_autoport-Gemma-4-26B-A4B-it_p300x2_release",
    "model_repo": "google/gemma-4-26B-A4B-it",
    "inference_engine": "vLLM",
    "tt_metal_commit": "4b17e185dea9e70db276bcd5b0ed47c5786738b0",
    "vllm_commit": "938c45ed71f3f669ffd38e4c9a033c3391cec961",
    "model_impl": "autoport-gemma4-26b-a4b-it"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (1/1 passed)
- Evals: 🟨 `NA` (0/2 passed, 2 NA)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for google/gemma-4-26B-A4B-it on P300X2

| Task          | Tolerance | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:--------------|:----------|:------|:-------------------|:-------------------|:---------------|
| meta_ifeval   | 0.05      | 82.62 | N/A                | N/A                | 🟨 NA          |
| meta_gpqa_cot | 0.05      | 40    | N/A                | N/A                | 🟨 NA          |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 267.5     | 264.2         | 294.8         | 35.8      | 4820.0    | 26.6              | 0.207          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1000        | 0.2675     | ✅ PASS    | 10.00            | 2.79            | ✅ PASS         | 10                 | 2.655             | ✅ PASS           |
| complete   | 500         | 0.5349     | ✅ PASS    | 20.00            | 1.395           | ✅ PASS         | 20                 | 1.328             | ✅ PASS           |
| target     | 300         | 0.8915     | ✅ PASS    | 26.00            | 1.073           | ✅ PASS         | 26                 | 1.021             | ✅ PASS           |

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
| Total Duration | 1026.75s                  |
| Total Attempts | 2                         |
| Generated      | 2026-08-16T09:38:13+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1026.74s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for google/gemma-4-26B-A4B-it on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for google/gemma-4-26B-A4B-it on P300X2

| Endpoint URL                              | model_name                | Task                  |
|:------------------------------------------|:--------------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | google/gemma-4-26B-A4B-it | vllm_chat_completions |

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
