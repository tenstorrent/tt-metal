## Tenstorrent Model Release Summary: tiiuae/Falcon3-7B-Base on P300X2

### Metadata: tiiuae/Falcon3-7B-Base on P300X2

```json
{
    "model_name": "tiiuae/Falcon3-7B-Base",
    "device": "P300X2",
    "generated_at": "2026-08-12T22:58:15+00:00",
    "report_id": "tiiuae__Falcon3-7B-Base_2026-08-12T225815+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release-cache/falcon3-base-release/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-12_22-50-29_id_autoport-Falcon3-7B-Base_p300x2_Qxbe2kf5.json",
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
- Evals: ✅ `PASS` (0/2 passed, 2 waived)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for tiiuae/Falcon3-7B-Base on P300X2

| Task                           | Tolerance | Published Score | Published Score Ref                                      | GPU Reference Score | gpu_reference_score_ref                                  | Score | Ratio to Published | Ratio to Reference | Accuracy Check | error                         |
|:-------------------------------|:----------|:----------------|:---------------------------------------------------------|:--------------------|:---------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:------------------------------|
| ifeval                         | 0.05      | 34.3            | https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks | 34.3                | https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks | 18.67 | 0.5443             | 0.5443             | ❌ FAIL        | N/A                           |
| gpqa_diamond_generative_n_shot | 0.05      | 35.5            | https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks | N/A                 | N/A                                                      | N/A   | N/A                | N/A                | ❌ FAIL        | no eval results parsed (rc=1) |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 200.8     | 178.5         | 336.2         | 16.1      | 2245.1    | 57.0              | 0.445          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 250         | 0.8031     | ✅ PASS    | 50.00            | 1.242           | ✅ PASS         | 50                 | 1.14              | ✅ PASS           |
| complete   | 250         | 0.8031     | ✅ PASS    | 50.00            | 1.242           | ✅ PASS         | 50                 | 1.14              | ✅ PASS           |
| target     | 250         | 0.8031     | ✅ PASS    | 50.00            | 1.242           | ✅ PASS         | 50                 | 1.14              | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 128   | 128  | 481.6     | 406.6         | 598.6         | 16.7      | 2604.4    | 1572.1            | 12.282         |
| 1           | 4            | 128   | 1024 | 222.6     | 185.0         | 337.6         | 15.1      | 15704.1   | 65.2              | 0.064          |
| 28          | 112          | 128   | 1024 | 844.1     | 580.0         | 1207.2        | 16.5      | 17735.5   | 1616.5            | 1.579          |
| 1           | 4            | 1024  | 128  | 2813.5    | 1364.5        | 6989.7        | 16.8      | 4943.5    | 25.9              | 0.202          |
| 28          | 112          | 1024  | 128  | 4301.0    | 2859.7        | 8813.3        | 20.6      | 6919.3    | 517.9             | 4.046          |
| 1           | 4            | 2048  | 128  | 3001.3    | 2973.3        | 3085.3        | 17.4      | 5214.7    | 24.5              | 0.192          |
| 15          | 60           | 2048  | 128  | 5850.5    | 6049.6        | 6059.4        | 20.6      | 8461.5    | 226.9             | 1.773          |
| 1           | 4            | 4096  | 128  | 6615.9    | 6465.4        | 7052.6        | 18.8      | 9003.4    | 14.2              | 0.111          |
| 7           | 28           | 4096  | 128  | 12228.8   | 12983.2       | 13786.8       | 26.6      | 15611.3   | 57.4              | 0.448          |
| 1           | 2            | 8192  | 128  | 15805.1   | 15805.1       | 16595.4       | 21.6      | 18552.4   | 6.9               | 0.054          |
| 3           | 6            | 8192  | 128  | 24991.7   | 29978.3       | 29984.5       | 60.9      | 32726.3   | 11.7              | 0.092          |
| 1           | 2            | 16384 | 128  | 39337.6   | 39337.6       | 40454.2       | 27.0      | 42770.5   | 3.0               | 0.023          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
