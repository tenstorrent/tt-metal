## Tenstorrent Model Release Summary: Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

### Metadata: Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

```json
{
    "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "device": "P300X2",
    "generated_at": "2026-08-19T00:28:52+00:00",
    "report_id": "Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3-Coder-30B-A3B-Instruct --impl qwen3-coder-30b-a3b-autoport --tt-device p300x2 --workflow release --dev-mode --server-url http://127.0.0.1:8100 --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-18_22-35-11_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2_ZIJqG3Un.json",
    "model_id": "id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2",
    "model_repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "qwen3-coder-30b-a3b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/19 passed, 19 NA)
- Evals: ✅ `PASS` (0/6 passed, 3 waived, 3 NA)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

| Task                      | Tolerance | Score | Ratio to Published | Ratio to Reference | Accuracy Check | error                         |
|:--------------------------|:----------|:------|:-------------------|:-------------------|:---------------|:------------------------------|
| mbpp_instruct             | 0.05      | 77.2  | N/A                | N/A                | 🟨 NA          | N/A                           |
| humaneval_instruct        | 0.05      | 92.68 | N/A                | N/A                | 🟨 NA          | N/A                           |
| meta_ifeval               | 0.05      | N/A   | N/A                | N/A                | ❌ FAIL        | no eval results parsed (rc=1) |
| meta_gpqa_cot             | 0.05      | N/A   | N/A                | N/A                | ❌ FAIL        | no eval results parsed (rc=1) |
| ifeval                    | 0.05      | 81.15 | N/A                | N/A                | 🟨 NA          | N/A                           |
| gpqa_diamond_cot_zeroshot | 0.05      | N/A   | N/A                | N/A                | ❌ FAIL        | no eval results parsed (rc=1) |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 128    | 128  | 401.2     | 401.0         | 407.1         | 230.2     | 29630.8   | 4.3               | 0.034          |
| 32          | 256          | 128    | 128  | 5366.4    | 5369.3        | 5401.7        | 261.2     | 38541.5   | 106.3             | 0.830          |
| 1           | 4            | 128    | 1024 | 406.1     | 407.5         | 407.9         | 230.5     | 236228.2  | 4.3               | 0.004          |
| 32          | 128          | 128    | 1024 | 5370.4    | 5376.2        | 5384.2        | 259.6     | 270926.6  | 120.9             | 0.118          |
| 1           | 4            | 1024   | 128  | 4981.4    | 1206.0        | 15862.1       | 230.9     | 34308.7   | 3.7               | 0.029          |
| 32          | 128          | 1024   | 128  | 30951.3   | 30958.0       | 30964.3       | 264.2     | 64504.7   | 63.5              | 0.496          |
| 1           | 4            | 2048   | 128  | 9422.6    | 2130.0        | 30427.5       | 230.0     | 38631.7   | 3.3               | 0.026          |
| 32          | 128          | 2048   | 128  | 60640.5   | 60640.6       | 60659.2       | 265.1     | 94303.5   | 43.4              | 0.339          |
| 1           | 4            | 4096   | 128  | 14308.8   | 4040.0        | 43885.5       | 231.0     | 43646.8   | 2.9               | 0.023          |
| 32          | 128          | 4096   | 128  | 121702.4  | 121711.1      | 121714.4      | 269.0     | 155870.0  | 26.3              | 0.205          |
| 1           | 2            | 8192   | 128  | 49906.7   | 49906.7       | 90926.1       | 232.6     | 79451.8   | 1.6               | 0.013          |
| 31          | 62           | 8192   | 128  | 242410.2  | 242273.8      | 242635.6      | 275.3     | 277379.6  | 14.3              | 0.112          |
| 1           | 2            | 16384  | 128  | 100171.7  | 100171.7      | 181824.4      | 236.3     | 130178.8  | 1.0               | 0.008          |
| 15          | 30           | 16384  | 128  | 249591.4  | 249456.8      | 249921.0      | 258.8     | 282457.2  | 6.8               | 0.053          |
| 1           | 1            | 32768  | 128  | 394207.7  | 394207.7      | 394207.7      | 243.4     | 425121.1  | 0.3               | 0.002          |
| 7           | 7            | 32768  | 128  | 263490.1  | 263532.2      | 263534.3      | 251.3     | 295404.9  | 3.0               | 0.024          |
| 1           | 1            | 65536  | 128  | 926825.9  | 926825.9      | 926825.9      | 258.9     | 959706.9  | 0.1               | 0.001          |
| 3           | 3            | 65536  | 128  | 276869.8  | 276956.1      | 276957.0      | 261.2     | 310044.3  | 1.2               | 0.010          |
| 1           | 1            | 131072 | 128  | 5662485.6 | 5662485.6     | 5662485.6     | 289.7     | 5699275.9 | 0.0               | 0.000          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
