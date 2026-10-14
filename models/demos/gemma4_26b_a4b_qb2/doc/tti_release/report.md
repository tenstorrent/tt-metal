## Tenstorrent Model Release Summary: google/gemma-4-26B-A4B-it on P300X2

### Metadata: google/gemma-4-26B-A4B-it on P300X2

```json
{
    "model_name": "gemma-4-26B-A4B-it",
    "model_repo": "google/gemma-4-26B-A4B-it",
    "device": "P300X2",
    "generated_at": "2026-10-13T10:38:37+00:00",
    "report_id": "id_tt-vllm-plugin_gemma-4-26B-A4B-it_p300x2_2026-10-13_11-17-57",
    "workflow": "release",
    "report_partial": false,
    "report_blocks": 16,
    "server_mode": "docker",
    "run_command": "python run.py --model google/gemma-4-26B-A4B-it --workflow release --device p300x2 --docker-server --dev-mode --impl tt-vllm-plugin --ci-mode --override-docker-image ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.23.0-78e87bd9380740a522ec4b5ab62a46c9498d7b35-0a76258-112700448945 --host-volume /home/ubuntu/actions-runner/_work/tt-shield/tt-shield/tt-inference-server/persistent-volume",
    "runtime_model_spec_json": "/home/ubuntu/actions-runner/_work/tt-shield/tt-shield/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-10-13_10-14-59_id_tt-vllm-plugin_gemma-4-26B-A4B-it_p300x2_t5d4psW9.json",
    "model_id": "id_tt-vllm-plugin_gemma-4-26B-A4B-it_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": "78e87bd9380740a522ec4b5ab62a46c9498d7b35",
    "vllm_commit": "0a76258",
    "model_impl": "tt-vllm-plugin"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (0/13 passed, 1 waived, 12 NA)
- Evals: ❌ `FAIL` (2/3 passed, 1 failed)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)

#### Blockers

- `evals:LLM Eval — swe_bench_verified`: Accuracy check failed.

---

### Accuracy Evaluations for google/gemma-4-26B-A4B-it on P300X2

| Task               | Tolerance | Published Score | Published Score Ref                                | GPU Reference Score | gpu_reference_score_ref | Score | Ratio to Published | Ratio to Reference | Accuracy Check | Accuracy Rule | Noise n | Noise SE | Noise z | scoring_error | priority | mean_seconds_per_task |
|:-------------------|:----------|:----------------|:---------------------------------------------------|:--------------------|:------------------------|:------|:-------------------|:-------------------|:---------------|:--------------|:--------|:---------|:--------|:--------------|:---------|:----------------------|
| r1_gpqa_diamond    | 0         | 82.3            | https://ai.google.dev/gemma/docs/core/model_card_4 | 80                  | [CI_NIGHTLY subset]     | 80    | 0.9721             | 1                  | ✅ PASS        | ratio         | 10      | 12.65    | 1.96    | false         | must     | 96.95                 |
| terminal_bench_2   | 0.05      | 14              | https://arxiv.org/abs/2607.02770                   | N/A                 | N/A                     | 40    | 2.857              | N/A                | ✅ PASS        | N/A           | N/A     | N/A      | N/A     | N/A           | must     | 2167                  |
| swe_bench_verified | 0.05      | N/A             | https://ai.google.dev/gemma/docs/core/model_card_4 | N/A                 | N/A                     | 0     | N/A                | N/A                | ❌ FAIL        | N/A           | N/A     | N/A      | N/A     | N/A           | must     | 7259                  |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 89.8      | 88.9          | 94.8          | 18.2      | 2397.7    | 53.4             | 53.4              | 106.8            | 0.417          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Output Target | Tput Output Ratio | Tput Output Check | TPOT Check | E2EL Check | tput_total Check | tput_input Check | goodput Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|:-----------|:-----------|:-----------------|:-----------------|:--------------|
| functional | 240         | 0.3742     | ✅ PASS    | 22.40            | 2.457           | ✅ PASS         | 22.4               | 2.383             | ✅ PASS           | 🟨 NA      | 🟨 NA      | 🟨 NA            | 🟨 NA            | 🟨 NA         |
| complete   | 48          | 1.871      | ❌ FAIL    | 112.00           | 0.4913          | ❌ FAIL         | 112                | 0.4766            | ❌ FAIL           | 🟨 NA      | 🟨 NA      | 🟨 NA            | 🟨 NA            | 🟨 NA         |
| target     | 24          | 3.742      | ❌ FAIL    | 224.00           | 0.2457          | ❌ FAIL         | 224                | 0.2383            | ❌ FAIL           | 🟨 NA      | 🟨 NA      | 🟨 NA            | 🟨 NA            | 🟨 NA         |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 1            | 261888 | 128  | 175991.4  | 175991.4      | 175991.4      | 21.2      | 178678.5  | 1465.7           | 0.7               | 1466.4           | 0.006          |
| 1           | 4            | 128    | 1024 | 94.2      | 94.2          | 95.0          | 18.4      | 18947.7   | 6.8              | 54.0              | 60.8             | 0.053          |
| 1           | 4            | 1024   | 128  | 496.2     | 494.8         | 503.5         | 18.7      | 2869.5    | 356.8            | 44.6              | 401.4            | 0.348          |
| 1           | 4            | 2048   | 128  | 991.1     | 990.1         | 994.5         | 18.7      | 3372.3    | 607.2            | 38.0              | 645.2            | 0.296          |
| 1           | 4            | 4096   | 128  | 1978.7    | 1981.5        | 1988.0        | 18.8      | 4363.8    | 938.6            | 29.3              | 967.9            | 0.229          |
| 1           | 2            | 8192   | 128  | 3958.9    | 3958.9        | 3973.1        | 18.8      | 6349.4    | 1290.1           | 20.2              | 1310.3           | 0.158          |
| 1           | 2            | 8192   | 1024 | 3960.4    | 3960.4        | 3976.0        | 18.9      | 23252.9   | 352.3            | 44.0              | 396.3            | 0.043          |
| 1           | 2            | 10000  | 1024 | 4936.9    | 4936.9        | 4954.6        | 18.9      | 24253.3   | 412.3            | 42.2              | 454.5            | 0.041          |
| 1           | 2            | 16384  | 128  | 8001.5    | 8001.5        | 8003.8        | 18.9      | 10403.4   | 1574.8           | 12.3              | 1587.1           | 0.096          |
| 1           | 1            | 32768  | 128  | 15780.2   | 15780.2       | 15780.2       | 19.1      | 18207.9   | 1799.6           | 7.0               | 1806.6           | 0.055          |
| 1           | 1            | 65536  | 128  | 33385.8   | 33385.8       | 33385.8       | 19.5      | 35866.9   | 1827.2           | 3.6               | 1830.7           | 0.028          |
| 1           | 1            | 131072 | 128  | 74394.7   | 74394.7       | 74394.7       | 20.2      | 76964.7   | 1703.0           | 1.7               | 1704.7           | 0.013          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
