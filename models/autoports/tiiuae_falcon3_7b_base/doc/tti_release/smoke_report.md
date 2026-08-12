## Tenstorrent Model Release Summary: tiiuae/Falcon3-7B-Base on P300X2

### Metadata: tiiuae/Falcon3-7B-Base on P300X2

```json
{
    "model_name": "tiiuae/Falcon3-7B-Base",
    "device": "P300X2",
    "generated_at": "2026-08-12T22:41:24+00:00",
    "report_id": "tiiuae__Falcon3-7B-Base_2026-08-12T224124+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --workflow benchmarks --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/autoport_smoke_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode smoke-test",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release-cache/falcon3-base-smoke/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-12_22-41-13_id_autoport-Falcon3-7B-Base_p300x2_-L5HNwAK.json",
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
- Benchmarks: ✅ `PASS` (1/1 passed)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 8 / OSL 8, concurrency 1 for tiiuae/Falcon3-7B-Base on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 1            | 8   | 8   | 194.4     | 194.4         | 194.4         | 45.3      | 511.8     | 15.6              | 1.952          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 6e+04       | 0.00324    | ✅ PASS    | 0.01             | 2206            | ✅ PASS         | 0.01               | 1562              | ✅ PASS           |
| complete   | 6e+04       | 0.00324    | ✅ PASS    | 0.01             | 2206            | ✅ PASS         | 0.01               | 1562              | ✅ PASS           |
| target     | 6e+04       | 0.00324    | ✅ PASS    | 0.01             | 2206            | ✅ PASS         | 0.01               | 1562              | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.
