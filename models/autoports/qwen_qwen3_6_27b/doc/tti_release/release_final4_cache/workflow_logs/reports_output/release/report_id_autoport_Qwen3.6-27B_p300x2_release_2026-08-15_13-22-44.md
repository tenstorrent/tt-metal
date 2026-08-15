## Tenstorrent Model Release Summary: Qwen/Qwen3.6-27B on P300X2

### Metadata: Qwen/Qwen3.6-27B on P300X2

```json
{
    "model_name": "Qwen/Qwen3.6-27B",
    "device": "P300X2",
    "generated_at": "2026-08-15T10:43:17+00:00",
    "report_id": "id_autoport_Qwen3.6-27B_p300x2_release_2026-08-15_13-22-44",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3.6-27B --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/tti_release/autoport_release_spec.json --tt-device p300x2 --workflow release --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/tti_release/release_final4_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-15_10-05-59_id_autoport_Qwen3.6-27B_p300x2_release_3pa-lnLO.json",
    "model_id": "id_autoport_Qwen3.6-27B_p300x2_release",
    "model_repo": "Qwen/Qwen3.6-27B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "f7119ed18595b9262528a55e87265173818ade0b",
    "vllm_commit": "c5f35e55071e8b5b3af7796e23ecc371a5859f24",
    "model_impl": "qwen36-blackhole"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/10 passed, 10 NA)
- Evals: ❌ `FAIL` (0/3 passed, 1 failed, 2 NA)
- Spec Tests: 🟨 `NA` (no blocks present)

#### Blockers

- `evals:LLM Eval — terminal_bench_2`: LLM Eval — terminal_bench_2 reported success=False (attempts=?)

---

### Accuracy Evaluations for Qwen/Qwen3.6-27B on P300X2

| Task             | eval_task_name            | Tolerance | Published Score | Published Score Ref                     | GPU Reference Score | Score | Ratio to Published | Ratio to Reference | Accuracy Check | Success | subprocess_rc |
|:-----------------|:--------------------------|:----------|:----------------|:----------------------------------------|:--------------------|:------|:-------------------|:-------------------|:---------------|:--------|:--------------|
| meta_ifeval      | ifeval                    | 0.05      | N/A             | N/A                                     | N/A                 | 26.37 | N/A                | N/A                | 🟨 NA          | N/A     | N/A           |
| meta_gpqa_cot    | gpqa_diamond_cot_zeroshot | 0.05      | N/A             | N/A                                     | N/A                 | 30    | N/A                | N/A                | 🟨 NA          | N/A     | N/A           |
| terminal_bench_2 | N/A                       | 0.05      | 59.3            | https://huggingface.co/Qwen/Qwen3.6-27B | 53.9                | N/A   | N/A                | N/A                | ❌ FAIL        | false   | 1             |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for Qwen/Qwen3.6-27B on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 128    | 128  | 4374.1    | 3921.0        | 5840.0        | 61.3      | 12162.1   | 10.5              | 0.082          |
| 1           | 4            | 128    | 1024 | 3939.8    | 3938.4        | 3962.7        | 57.4      | 62624.4   | 16.4              | 0.016          |
| 1           | 4            | 1024   | 128  | 32556.3   | 33247.5       | 34944.6       | 61.5      | 40372.8   | 3.2               | 0.025          |
| 1           | 4            | 2048   | 128  | 65851.0   | 68586.4       | 68958.5       | 61.6      | 73671.8   | 1.7               | 0.014          |
| 1           | 4            | 4096   | 128  | 128605.8  | 132967.1      | 134349.7      | 61.7      | 136446.3  | 0.9               | 0.007          |
| 1           | 2            | 8192   | 128  | 266138.6  | 266138.6      | 267104.3      | 62.0      | 274007.4  | 0.5               | 0.004          |
| 1           | 2            | 16384  | 128  | 539901.2  | 539901.2      | 543975.3      | 62.3      | 547808.9  | 0.2               | 0.002          |
| 1           | 1            | 32768  | 128  | 912692.5  | 912692.5      | 912692.5      | 68.3      | 921366.0  | 0.1               | 0.001          |
| 1           | 1            | 65536  | 128  | 1857210.3 | 1857210.3     | 1857210.3     | 69.4      | 1866025.0 | 0.1               | 0.001          |
| 1           | 1            | 131072 | 128  | 3706685.4 | 3706685.4     | 3706685.4     | 70.9      | 3715685.0 | 0.0               | 0.000          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.