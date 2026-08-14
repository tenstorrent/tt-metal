## Tenstorrent Model Release Summary: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

### Metadata: mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

```json
{
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "device": "P300X2",
    "generated_at": "2026-08-14T17:51:45+00:00",
    "report_id": "autoport_mistral_small_24b_p300x2_v9_benchmark_repaired",
    "workflow": "benchmarks-supplemental",
    "scope": "Regraded preserved 8-request benchmark evidence only; not a rerun or aggregate release verdict.",
    "source_evidence": "benchmark_raw_v9_fixed.json",
    "model_status": "FUNCTIONAL",
    "server_mode": "API",
    "tt_metal_commit_at_v9": "1529e332a1c37937a682ba04b77e7dc3418f2589",
    "vllm_commit_at_v9": "6bd775d4f3a41d09d3ed03c40b45b5f9621fff9e"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: ✅ `PASS` (1/1 passed)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for mistralai/Mistral-Small-24B-Instruct-2501 on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 127 | 128 | 1272.7    | 1273.3        | 1276.2        | 19.2      | 3709.7    | 34.5              | 0.270          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1400        | 0.9091     | ✅ PASS    | 50.00            | 1.042           | ✅ PASS         | 32                 | 1.078             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects the strictest configured tier with at least one measurable check. The Target Checks table shows each configured tier — functional, complete, and/or target — from most to least lenient.
