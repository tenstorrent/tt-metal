## Tenstorrent Model Release Summary: meta-models/Muse-Glimmer-30B on P300X2

### Metadata: meta-models/Muse-Glimmer-30B on P300X2

```json
{
    "model_name": "meta-models/Muse-Glimmer-30B",
    "device": "P300X2",
    "generated_at": "2026-08-16T09:36:35+00:00",
    "report_id": "id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_2026-08-16_06-05-37",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Muse-Glimmer-30B --runtime-model-spec-json /home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/specs/muse_glimmer_30b_autoport_release.json --tt-device p300x2 --workflow release --service-port 8000 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/cache_root/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-16_04-40-51_id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2_jOKQ172f.json",
    "model_id": "id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2",
    "model_repo": "meta-models/Muse-Glimmer-30B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "7db0eca",
    "vllm_commit": null,
    "model_impl": "muse-glimmer-30b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/18 passed, 17 NA)
- Evals: ✅ `PASS` (2/2 passed)
- Spec Tests: ✅ `PASS` (0/1 passed, 1 waived)
- All acceptance criteria passed.

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
| 1           | 8            | 127 | 128 | 72.0      | 72.5          | 75.4          | 23.0      | 2998.0    | 42.7              | 0.334          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 88.3        | 0.8151     | ✅ PASS    | 11.33            | 3.831           | ✅ PASS         | 11.33              | 3.768             | ✅ PASS           |
| complete   | 17.66       | 4.075      | ❌ FAIL    | 56.65            | 0.7662          | ❌ FAIL         | 56.65              | 0.7536            | ❌ FAIL           |
| target     | 8.83        | 8.151      | ❌ FAIL    | 113.30           | 0.3831          | ❌ FAIL         | 113.3              | 0.3768            | ❌ FAIL           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for meta-models/Muse-Glimmer-30B on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 127   | 128  | 2017.8    | 2014.1        | 2060.2        | 23.1      | 4950.1    | 827.3             | 6.463          |
| 1           | 4            | 127   | 1024 | 77.5      | 78.3          | 80.1          | 23.5      | 24086.6   | 42.5              | 0.042          |
| 32          | 128          | 127   | 1024 | 2024.0    | 2036.9        | 2043.1        | 24.0      | 26610.7   | 1231.3            | 1.202          |
| 1           | 4            | 1023  | 128  | 154.2     | 154.2         | 155.5         | 23.9      | 3183.6    | 40.2              | 0.314          |
| 32          | 128          | 1023  | 128  | 4695.5    | 4693.6        | 4736.2        | 24.7      | 7835.1    | 522.7             | 4.083          |
| 1           | 4            | 2047  | 128  | 267.5     | 267.6         | 269.7         | 24.2      | 3340.1    | 38.3              | 0.299          |
| 32          | 128          | 2047  | 128  | 8191.8    | 8199.4        | 8209.7        | 25.7      | 11455.3   | 357.5             | 2.793          |
| 1           | 4            | 4095  | 128  | 489.7     | 489.8         | 495.7         | 24.5      | 3599.1    | 35.6              | 0.278          |
| 32          | 128          | 4095  | 128  | 15181.8   | 15154.4       | 15262.9       | 26.1      | 18499.7   | 221.4             | 1.730          |
| 1           | 2            | 8191  | 128  | 979.5     | 979.5         | 987.2         | 24.9      | 4143.3    | 30.9              | 0.241          |
| 32          | 64           | 8191  | 128  | 30353.1   | 30806.0       | 30910.9       | 30.6      | 34233.2   | 119.6             | 0.935          |
| 1           | 2            | 16383 | 128  | 2188.0    | 2188.0        | 2195.4        | 25.8      | 5469.9    | 23.4              | 0.183          |
| 32          | 64           | 16383 | 128  | 55679.8   | 53985.3       | 69169.3       | 135.6     | 72902.8   | 56.2              | 0.439          |
| 1           | 1            | 32767 | 128  | 4729.4    | 4729.4        | 4729.4        | 27.7      | 8243.2    | 15.5              | 0.121          |
| 31          | 31           | 32767 | 128  | 98363.7   | 98151.0       | 147361.2      | 419.9     | 151696.8  | 26.2              | 0.204          |
| 1           | 1            | 65535 | 128  | 10768.1   | 10768.1       | 10768.1       | 31.5      | 14762.3   | 8.7               | 0.068          |
| 16          | 16           | 65535 | 128  | 117927.8  | 120947.8      | 179000.6      | 521.0     | 184097.9  | 10.4              | 0.081          |

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
| Total Duration | 367.02s                   |
| Total Attempts | 2                         |
| Generated      | 2026-08-16T09:36:35+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ❌ FAIL | VLLMParamConformanceTest | 367.01s  | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for meta-models/Muse-Glimmer-30B on P300X2

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for meta-models/Muse-Glimmer-30B on P300X2

| Endpoint URL                              | model_name                   | Task                  | known_issue_waiver                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:------------------------------------------|:-----------------------------|:----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| http://127.0.0.1:8000/v1/chat/completions | meta-models/Muse-Glimmer-30B | vllm_chat_completions | test_penalties: test_penalties[presence_penalty-1.2-repeat_trap] asserts unique_ratio(penalty) >= unique_ratio(base) * 0.90; this autoport measures 0.8207 deterministically (3/3 identical texts per arm). The row is not a property of the penalty implementation. Rebuilding vLLM's own rule argmax(raw_logprob - 1.2*[token already generated]) from raw logprobs reproduces the device's emitted tokens 160/160, while penalise-by-count, penalise-prompt-tokens-too and no-penalty first contradict it at steps 30, 88 and 9; and running this row's own comparison through vLLM's float32 host sampler - no Tenstorrent sampling code in the path at all - fails the same assertion more often and more severely than the device does (1 pass/4 vs 2 pass/4; 0.3585 vs 0.9725 on the greedy, RNG-free trial). presence_penalty is a flat one-shot offset that does not grow with repetition count, so on the "Write a very repetitive story." trap prompt it cannot break a sentence-level loop, only move the model into a different one - which is why llm_module/test_vllm_chat_completions.py:313 already exempts presence_penalty from the stronger "heavy repetition should decrease" assertion on this exact prompt. Same waiver as the canonical Llama-3.3-70B-Instruct P300X2 entry, tracked in tenstorrent/tt-inference-server#3888. Full evidence: models/autoports/meta_models_muse_glimmer_30b/doc/tti_release/AUTOFIX_presence_penalty.md |

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
