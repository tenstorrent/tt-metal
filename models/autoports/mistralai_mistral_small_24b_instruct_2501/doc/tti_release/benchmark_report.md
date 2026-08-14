## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-14T17:51:45+00:00",
    "report_id": "mistralai__Mistral-Small-24B-Instruct-2501_2026-08-14T175145+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run_workflows.py --model Mistral-Small-24B-Instruct-2501 --workflow benchmarks --device p300x2 --service-port 8000 --server-url http://127.0.0.1 --tools vllm --runtime-model-spec-json /home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_15-56-55_id_autoport_mistral-small-24b-instruct-2501_p300x2_7efNyisP.json --output-dir /home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9_benchmark_fixed",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/tti_cache_release_v9/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_15-56-55_id_autoport_mistral-small-24b-instruct-2501_p300x2_7efNyisP.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "1529e332a1c37937a682ba04b77e7dc3418f2589",
    "vllm_commit": "6bd775d4f3a41d09d3ed03c40b45b5f9621fff9e",
    "model_impl": "mistral-small-24b-2501-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 127 | 128 | 1272.7    | 1273.3        | 1276.2        | 19.2      | 3709.7    | 34.5              | 0.270          |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.9091     | ✅ PASS    | 50.00            | 1.042           | ✅ PASS         | 32                 | 1.078             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
