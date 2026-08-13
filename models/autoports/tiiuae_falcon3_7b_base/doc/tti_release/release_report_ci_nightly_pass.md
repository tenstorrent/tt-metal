## Tenstorrent Model Release Summary: tiiuae/Falcon3-7B-Base on P300X2

### Metadata: tiiuae/Falcon3-7B-Base on P300X2

```json
{
    "model_name": "tiiuae/Falcon3-7B-Base",
    "device": "P300X2",
    "generated_at": "2026-08-13T10:50:32+00:00",
    "report_id": "tiiuae__Falcon3-7B-Base_2026-08-13T105032+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly --disable-trace-capture",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release-cache/falcon3-base-stage11-contextfix-final/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-13_10-49-53_id_autoport-Falcon3-7B-Base_p300x2_Kpm7-0VD.json",
    "model_id": "id_autoport-Falcon3-7B-Base_p300x2",
    "model_repo": "tiiuae/Falcon3-7B-Base",
    "inference_engine": "vLLM",
    "tt_metal_commit": "053fb3f6362189a6fae76632143ec8faa569f532",
    "vllm_commit": "7c99bd3b8",
    "model_impl": "autoport-falcon3-7b-base"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (1/13 passed, 12 NA)
- Evals: ✅ `PASS` (2/2 passed)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for tiiuae/Falcon3-7B-Base on P300X2

| Task                           | Tolerance | Published Score | Published Score Ref                                      | GPU Reference Score | gpu_reference_score_ref                                                                                                                                                                    | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:-------------------------------|:----------|:----------------|:---------------------------------------------------------|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|
| ifeval                         | 0.05      | 34.3            | https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks | 17.86               | paired CPU HF control: exact Falcon3-7B-Base snapshot bf3d7ed, lm-eval ifeval v4, raw prompts, seed 42, doc_ids 0-27 (5/28), 2026-08-13 [CI_NIGHTLY subset]                                | 21.43 | 0.6247             | 1.2                | ✅ PASS        |
| gpqa_diamond_generative_n_shot | 0.05      | 35.5            | https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks | 50                  | paired CPU HF control: exact Falcon3-7B-Base snapshot bf3d7ed, lm-eval gpqa_diamond_generative_n_shot v2, raw prompts, 0-shot, seed 42, doc_ids 0-9 (5/10), 2026-08-13 [CI_NIGHTLY subset] | 60    | 1.69               | 1.2                | ✅ PASS        |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 200.8     | 179.1         | 332.7         | 16.1      | 2247.3    | 57.0              | 0.445          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 250         | 0.8032     | ✅ PASS    | 50.00            | 1.241           | ✅ PASS         | 50                 | 1.139             | ✅ PASS           |
| complete   | 250         | 0.8032     | ✅ PASS    | 50.00            | 1.241           | ✅ PASS         | 50                 | 1.139             | ✅ PASS           |
| target     | 250         | 0.8032     | ✅ PASS    | 50.00            | 1.241           | ✅ PASS         | 50                 | 1.139             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 128   | 128  | 483.1     | 419.6         | 614.1         | 16.7      | 2607.8    | 1570.1            | 12.266         |
| 1           | 4            | 128   | 1024 | 224.4     | 188.1         | 337.0         | 15.1      | 15697.1   | 65.2              | 0.064          |
| 28          | 112          | 128   | 1024 | 513.2     | 401.9         | 738.9         | 16.5      | 17372.0   | 1650.4            | 1.612          |
| 1           | 4            | 1024  | 128  | 1377.8    | 1363.6        | 1419.5        | 16.8      | 3509.2    | 36.5              | 0.285          |
| 28          | 112          | 1024  | 128  | 2823.2    | 2846.5        | 3109.1        | 19.6      | 5314.5    | 674.2             | 5.267          |
| 1           | 4            | 2048  | 128  | 3002.1    | 2976.9        | 3081.8        | 17.4      | 5217.7    | 24.5              | 0.192          |
| 15          | 60           | 2048  | 128  | 5448.5    | 6044.2        | 6059.9        | 23.6      | 8451.7    | 227.1             | 1.775          |
| 1           | 4            | 4096  | 128  | 6465.6    | 6461.0        | 6484.7        | 18.8      | 8855.4    | 14.5              | 0.113          |
| 7           | 28           | 4096  | 128  | 12113.4   | 12993.3       | 13217.1       | 26.5      | 15478.2   | 57.9              | 0.452          |
| 1           | 2            | 8192  | 128  | 14988.2   | 14988.2       | 15000.8       | 21.6      | 17733.7   | 7.2               | 0.056          |
| 3           | 6            | 8192  | 128  | 24992.0   | 29977.5       | 29983.4       | 60.9      | 32730.7   | 11.7              | 0.092          |
| 1           | 2            | 16384 | 128  | 38235.3   | 38235.3       | 38247.7       | 26.9      | 41651.1   | 3.1               | 0.024          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
