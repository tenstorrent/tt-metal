## Tenstorrent Model Release Summary: Qwen/Qwen3-4B on P150X4

### Metadata: Qwen/Qwen3-4B on P150X4
```json
{
    "model_name": "Qwen/Qwen3-4B",
    "device": "P150X4",
    "generated_at": "2026-07-03T07:28:31+00:00",
    "report_id": "Qwen__Qwen3-4B_2026-07-03T072831+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --model Qwen3-4B --runtime-model-spec-json /home/ubuntu/tti-release/qwen_qwen3_4b_20260703/autoport_smoke_runtime_model_spec.json --tt-device p150x4 --workflow benchmarks --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture",
    "runtime_model_spec_json": "/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/smoke_tti_cache_target_only/workflow_logs/runtime_model_specs/runtime_model_spec_2026-07-03_07-28-15_id_tt-vllm-plugin_Qwen3-4B_p150x4_autoport_smoke_oyId73YE.json",
    "model_id": "id_tt-vllm-plugin_Qwen3-4B_p150x4_autoport_smoke",
    "model_repo": "Qwen/Qwen3-4B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "affc17f0d3bba0388b27e8fbc8853ea0aef3e421",
    "vllm_commit": "de6c44fd89154bd800c8c947e7205876b93013e3",
    "model_impl": "tt-vllm-plugin"
}
```

### Acceptance Criteria

- Acceptance status: `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: `NA` (no blocks present)
- Evals: `NA` (no blocks present)
- Spec Tests: `NA` (no blocks present)
- All acceptance criteria passed.

### Vllm for Qwen/Qwen3-4B on P150X4

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|-------------|--------------|-----|-----|-----------|---------------|---------------|-----------|-----------|-------------------|----------------|
|           1 |            1 |   8 |   8 |      93.2 |          93.2 |          93.2 |      59.4 |     509.3 |              15.7 |          1.961 |
