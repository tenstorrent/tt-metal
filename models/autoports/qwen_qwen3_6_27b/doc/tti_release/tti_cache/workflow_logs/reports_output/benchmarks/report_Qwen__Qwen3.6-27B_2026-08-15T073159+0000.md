## Tenstorrent Model Release Summary: Qwen/Qwen3.6-27B on P300X2

### Metadata: Qwen/Qwen3.6-27B on P300X2

```json
{
    "model_name": "Qwen/Qwen3.6-27B",
    "device": "P300X2",
    "generated_at": "2026-08-15T07:31:59+00:00",
    "report_id": "Qwen__Qwen3.6-27B_2026-08-15T073159+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3.6-27B --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/tti_release/autoport_smoke_spec.json --tt-device p300x2 --workflow benchmarks --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/tti_release/tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-15_07-31-50_id_autoport_Qwen3.6-27B_p300x2_smoke_seDiumYS.json",
    "model_id": "id_autoport_Qwen3.6-27B_p300x2_smoke",
    "model_repo": "Qwen/Qwen3.6-27B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "f7119ed18595b9262528a55e87265173818ade0b",
    "vllm_commit": "c5f35e55071e8b5b3af7796e23ecc371a5859f24",
    "model_impl": "qwen36-blackhole"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark for Qwen/Qwen3.6-27B on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 1            | 8   | 8   | 696.3     | 696.3         | 696.3         | 152.6     | 1764.5    | 4.5               | 0.567          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.