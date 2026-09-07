## Tenstorrent Model Release Summary: zai-org/GLM-4.7-Flash on P150

### Metadata: zai-org/GLM-4.7-Flash on P150

```json
{
    "model_name": "zai-org/GLM-4.7-Flash",
    "device": "P150",
    "generated_at": "2026-09-07T04:23:34+00:00",
    "report_id": "zai-org__GLM-4.7-Flash_2026-09-07T042334+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --model GLM-4.7-Flash --tt-device p150 --workflow benchmarks --service-port 8000 --dev-mode --no-auth --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/stisi/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-06_20-39-08_id_autoport-glm47-flash_GLM-4.7-Flash_p150_CDpwgIUy.json",
    "model_id": "id_autoport-glm47-flash_GLM-4.7-Flash_p150",
    "model_repo": "zai-org/GLM-4.7-Flash",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "autoport-glm47-flash"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (1/23 passed, 22 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for zai-org/GLM-4.7-Flash on P150

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 296.2     | 300.4         | 310.4         | 29.5      | 4041.8    | 31.7             | 31.7              | 63.3             | 0.247          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Output Target | Tput Output Ratio | Tput Output Check | TTFT Target | TTFT Ratio |
|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|:------------|:-----------|
| functional | 🟨 NA      | 25.64            | 1.322           | ✅ PASS         | 25.64              | 1.235             | ✅ PASS           | N/A         | N/A        |
| complete   | 🟨 NA      | 128.20           | 0.2645          | ❌ FAIL         | 128.2              | 0.247             | ❌ FAIL           | N/A         | N/A        |
| target     | ❌ FAIL    | 33.90            | 1               | ✅ PASS         | 31.84              | 0.9946            | ✅ PASS           | 274.1       | 1.081      |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for zai-org/GLM-4.7-Flash on P150

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 32          | 256          | 128    | 128  | 9314.0    | 9324.8        | 9388.0        | 91.6      | 20948.8   | 195.5            | 195.5             | 391.0            | 1.528          |
| 1           | 4            | 128    | 1024 | 290.3     | 290.0         | 303.1         | 30.7      | 31717.0   | 4.0              | 32.3              | 36.3             | 0.032          |
| 32          | 128          | 128    | 1024 | 9344.0    | 9356.5        | 9385.9        | 99.0      | 110634.8  | 37.0             | 296.2             | 333.2            | 0.289          |
| 1           | 4            | 1024   | 128  | 1961.2    | 1961.8        | 2044.9        | 31.4      | 5953.1    | 172.0            | 21.5              | 193.5            | 0.168          |
| 32          | 128          | 1024   | 128  | 64245.4   | 64249.7       | 64437.2       | 93.6      | 76134.2   | 430.4            | 53.8              | 484.2            | 0.420          |
| 1           | 4            | 2048   | 128  | 4043.9    | 4062.3        | 4173.8        | 33.6      | 8312.8    | 246.4            | 15.4              | 261.8            | 0.120          |
| 32          | 128          | 2048   | 128  | 131438.7  | 131459.6      | 131927.6      | 96.4      | 143679.9  | 456.1            | 28.5              | 484.6            | 0.223          |
| 1           | 4            | 4096   | 128  | 8496.7    | 8559.7        | 8702.9        | 37.2      | 13226.0   | 309.7            | 9.7               | 319.4            | 0.076          |
| 32          | 128          | 4096   | 128  | 274934.2  | 275316.7      | 275897.2      | 99.8      | 287611.9  | 455.7            | 14.2              | 470.0            | 0.111          |
| 1           | 2            | 8192   | 128  | 18155.2   | 18155.2       | 18423.8       | 45.2      | 23895.9   | 342.8            | 5.4               | 348.2            | 0.042          |
| 24          | 48           | 8192   | 128  | 449097.6  | 449072.3      | 450527.6      | 103.4     | 462235.6  | 425.3            | 6.6               | 432.0            | 0.052          |
| 1           | 2            | 8192   | 1024 | 18151.5   | 18151.5       | 18421.4       | 46.0      | 65241.2   | 125.6            | 15.7              | 141.3            | 0.015          |
| 22          | 44           | 8192   | 1024 | 411255.3  | 411232.1      | 412557.9      | 108.1     | 521868.1  | 345.3            | 43.2              | 388.5            | 0.042          |
| 1           | 2            | 10000  | 1024 | 23510.4   | 23510.4       | 23882.5       | 49.5      | 74134.8   | 134.9            | 13.8              | 148.7            | 0.013          |
| 18          | 36           | 10000  | 1024 | 434427.2  | 434403.7      | 435709.6      | 106.2     | 543050.7  | 331.5            | 33.9              | 365.4            | 0.033          |
| 1           | 2            | 16384  | 128  | 42467.7   | 42467.7       | 43012.7       | 60.4      | 50137.1   | 326.8            | 2.6               | 329.3            | 0.020          |
| 12          | 24           | 16384  | 128  | 522351.6  | 522328.0      | 523874.1      | 105.5     | 535744.4  | 367.0            | 2.9               | 369.8            | 0.022          |
| 1           | 1            | 32768  | 128  | 108669.3  | 108669.3      | 108669.3      | 91.5      | 120294.2  | 272.4            | 1.1               | 273.5            | 0.008          |
| 6           | 6            | 32768  | 128  | 668523.2  | 668531.8      | 668533.7      | 111.5     | 682687.9  | 288.0            | 1.1               | 289.1            | 0.009          |
| 1           | 1            | 65536  | 128  | 319521.2  | 319521.2      | 319521.2      | 153.1     | 338965.3  | 193.3            | 0.4               | 193.7            | 0.003          |
| 3           | 3            | 65536  | 128  | 952987.1  | 953002.3      | 953003.5      | 164.5     | 973878.4  | 201.9            | 0.4               | 202.3            | 0.003          |
| 1           | 1            | 131072 | 128  | 1033048.6 | 1033048.6     | 1033048.6     | 276.8     | 1068203.1 | 122.7            | 0.1               | 122.8            | 0.001          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
