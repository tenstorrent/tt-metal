## Tenstorrent Model Release Summary: tiiuae/Falcon3-7B-Base on P300X2

### Metadata: tiiuae/Falcon3-7B-Base on P300X2

```json
{
    "model_name": "tiiuae/Falcon3-7B-Base",
    "device": "P300X2",
    "generated_at": "2026-08-13T10:28:28+00:00",
    "report_id": "tiiuae__Falcon3-7B-Base_2026-08-13T102828+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly --disable-trace-capture",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release-cache/falcon3-base-stage11-final/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-13_10-27-49_id_autoport-Falcon3-7B-Base_p300x2_v3cF9JIW.json",
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
| 1           | 8            | 128 | 128 | 206.4     | 180.2         | 366.5         | 16.1      | 2249.4    | 56.9              | 0.445          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 250         | 0.8255     | ✅ PASS    | 50.00            | 1.243           | ✅ PASS         | 50                 | 1.138             | ✅ PASS           |
| complete   | 250         | 0.8255     | ✅ PASS    | 50.00            | 1.243           | ✅ PASS         | 50                 | 1.138             | ✅ PASS           |
| target     | 250         | 0.8255     | ✅ PASS    | 50.00            | 1.243           | ✅ PASS         | 50                 | 1.138             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 128   | 128  | 569.6     | 411.1         | 1368.3        | 16.8      | 2707.0    | 1512.6            | 11.817         |
| 1           | 4            | 128   | 1024 | 223.8     | 187.1         | 336.3         | 15.1      | 15690.4   | 65.3              | 0.064          |
| 28          | 112          | 128   | 1024 | 663.6     | 574.7         | 1178.4        | 16.5      | 17560.5   | 1632.6            | 1.594          |
| 1           | 4            | 1024  | 128  | 1377.3    | 1361.9        | 1422.7        | 16.8      | 3509.1    | 36.5              | 0.285          |
| 28          | 112          | 1024  | 128  | 2858.4    | 2843.1        | 3113.3        | 19.3      | 5315.0    | 674.2             | 5.267          |
| 1           | 4            | 2048  | 128  | 2992.0    | 2975.1        | 3047.1        | 17.4      | 5206.3    | 24.6              | 0.192          |
| 15          | 60           | 2048  | 128  | 5793.5    | 6041.0        | 6054.8        | 20.9      | 8452.1    | 227.1             | 1.774          |
| 1           | 4            | 4096  | 128  | 6472.6    | 6469.3        | 6492.2        | 18.8      | 8859.6    | 14.4              | 0.113          |
| 7           | 28           | 4096  | 128  | 11190.4   | 12986.0       | 13226.0       | 33.8      | 15478.6   | 57.9              | 0.452          |
| 1           | 2            | 8192  | 128  | 15010.6   | 15010.6       | 15019.4       | 21.6      | 17758.1   | 7.2               | 0.056          |
| 3           | 6            | 8192  | 128  | 22512.3   | 22515.3       | 29993.5       | 80.5      | 32736.6   | 11.7              | 0.092          |
| 1           | 2            | 16384 | 128  | 38243.9   | 38243.9       | 38254.1       | 26.9      | 41656.4   | 3.1               | 0.024          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
