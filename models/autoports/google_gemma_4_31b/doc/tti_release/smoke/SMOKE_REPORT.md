## Tenstorrent Model Release Summary: google/gemma-4-31B on P150X4

### Metadata: google/gemma-4-31B on P150X4

```json
{
    "model_name": "google/gemma-4-31B",
    "device": "P150X4",
    "generated_at": "2026-07-16T10:01:15+00:00",
    "report_id": "google__gemma-4-31B_2026-07-16T100115+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --model gemma-4-31B-it --runtime-model-spec-json ../autoport_smoke_spec.json --tt-device p150x4 --workflow benchmarks --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test",
    "runtime_model_spec_json": "/localdev/odjuricic/tt-metal/.exp_run/tti-release/gemma4-31b-20260716/smoke_cache_pass/workflow_logs/runtime_model_specs/runtime_model_spec_2026-07-16_10-01-01_id_autoport_google_gemma_4_31b_p150x4_smoke_STEunLYf.json",
    "model_id": "id_autoport_google_gemma_4_31b_p150x4_smoke",
    "model_repo": "google/gemma-4-31B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "2be0f245e20",
    "vllm_commit": "44b7853",
    "model_impl": "autoport-google-gemma-4-31b"
}
```

### Acceptance Criteria

- Acceptance status: `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: `NA` (no blocks present)
- Evals: `NA` (no blocks present)
- Spec Tests: `NA` (no blocks present)
- All acceptance criteria passed.

---

### Vllm for google/gemma-4-31B on P150X4

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|-------------|--------------|-----|-----|-----------|---------------|---------------|-----------|-----------|-------------------|----------------|
|           1 |            1 |   7 |   8 |     903.2 |         903.2 |         903.2 |     180.0 |    2163.4 |               3.7 |          0.462 |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.
