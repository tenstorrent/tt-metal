## Tenstorrent Model Release Summary: Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

### Metadata: Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

```json
{
    "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "device": "P300X2",
    "generated_at": "2026-08-18T19:40:37+00:00",
    "report_id": "Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-18T194037+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3-Coder-30B-A3B-Instruct --impl qwen3-coder-30b-a3b-autoport --tt-device p300x2 --workflow benchmarks --dev-mode --server-url http://127.0.0.1:8100 --service-port 8100 --limit-samples-mode smoke-test --disable-trace-capture --no-auth --skip-system-sw-validation",
    "runtime_model_spec_json": "/home/raahem/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-18_19-40-18_id_qwen3-coder-30b-a3b-autoport_Qwen3-Coder-30B-A3B-Instruct_p300x2_B91Prs82.json",
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
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark for Qwen/Qwen3-Coder-30B-A3B-Instruct on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 16  | 4   | 296.2     | 294.8         | 316.1         | 230.2     | 986.7     | 4.1               | 1.013          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
