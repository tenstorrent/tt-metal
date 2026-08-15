## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-15T01:51:46+00:00",
    "report_id": "mistralai__Mistral-Small-24B-Instruct-2501_2026-08-15T015146+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --workflow benchmarks --runtime-model-spec-json /home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/release_spec.json --tt-device p300x2 --no-auth --skip-system-sw-validation --disable-trace-capture --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/mvasiljevic/tti-release/mistral-small-24b-2501/authoritative_v10/component_cache_slotfix/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-14_23-10-45_id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3_PBq29y0H.json",
    "model_id": "id_autoport_mistral-small-24b-instruct-2501_p300x2_release_v10_lookahead3",
    "model_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
    "inference_engine": "vLLM",
    "tt_metal_commit": "5bab286dc7fb063f4f435c840af64359fe4bf533",
    "vllm_commit": "aab6d846caf95c5e9cf8038f3338650a9132c383",
    "model_impl": "mistral-small-24b-2501-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/13 passed, 12 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 127 | 128 | 535.4     | 535.4         | 536.3         | 18.9      | 2940.2    | 43.5              | 0.340          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.3824     | ✅ PASS    | 50.00            | 1.056           | ✅ PASS         | 32                 | 1.36              | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects the strictest configured tier with at least one measurable check. The Target Checks table shows each configured tier — functional, complete, and/or target — from most to least lenient.

---

### vLLM Benchmark for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 32          | 256          | 127   | 128  | 1231.9    | 1173.2        | 1927.2        | 19.5      | 3704.2    | 1105.5            | 8.637          |
| 1           | 4            | 127   | 1024 | 539.1     | 536.4         | 548.7         | 18.4      | 19409.2   | 52.8              | 0.051          |
| 28          | 112          | 127   | 1024 | 1318.5    | 1166.0        | 1902.7        | 18.9      | 20612.0   | 1391.0            | 1.358          |
| 1           | 4            | 1023  | 128  | 25238.0   | 25244.3       | 25291.6       | 20.0      | 27784.1   | 4.6               | 0.036          |
| 28          | 112          | 1023  | 128  | 49765.1   | 50517.2       | 51133.6       | 27.8      | 53293.5   | 67.2              | 0.525          |
| 1           | 4            | 2047  | 128  | 77666.3   | 77630.1       | 77906.8       | 21.4      | 80387.3   | 1.6               | 0.012          |
| 15          | 60           | 2047  | 128  | 150079.8  | 155188.9      | 155342.0      | 62.2      | 157978.5  | 12.2              | 0.095          |
| 1           | 4            | 4095  | 128  | 182447.9  | 182430.3      | 182738.3      | 24.0      | 185500.6  | 0.7               | 0.005          |
| 7           | 28           | 4095  | 128  | 339133.5  | 365065.6      | 365758.6      | 229.6     | 368295.1  | 2.4               | 0.019          |
| 1           | 2            | 8191  | 128  | 393025.0  | 393025.0      | 393085.3      | 29.2      | 396737.5  | 0.3               | 0.003          |
| 3           | 6            | 8191  | 128  | 524311.3  | 393240.6      | 786999.7      | 545.2     | 593555.0  | 0.6               | 0.005          |
| 1           | 2            | 16383 | 128  | 814523.9  | 814523.9      | 814762.0      | 39.7      | 819560.8  | 0.2               | 0.001          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
