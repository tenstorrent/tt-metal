## Tenstorrent Model Release Summary: google/gemma-4-26B-A4B-it on P300X2

### Metadata: google/gemma-4-26B-A4B-it on P300X2

```json
{
    "model_name": "gemma-4-26B-A4B-it",
    "model_repo": "google/gemma-4-26B-A4B-it",
    "device": "P300X2",
    "generated_at": "2026-09-29T14:42:47+00:00",
    "report_id": "google__gemma-4-26B-A4B-it_2026-09-29T144247+0000",
    "workflow": "benchmarks",
    "report_partial": false,
    "report_blocks": 5,
    "server_mode": "docker",
    "run_command": "python run.py --model google/gemma-4-26B-A4B-it --workflow benchmarks --device p300x2 --docker-server --dev-mode --impl gemma4-autoport --ci-mode --override-docker-image ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:461668dba77e062db3a1dae6743f63dcb8ba1c9b5f88b582c332a5661c680e6f",
    "runtime_model_spec_json": "/home/ubuntu/actions-runner/_work/tt-agentic-bringup-qb2/tt-agentic-bringup-qb2/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-29_14-24-31_id_gemma4-autoport_gemma-4-26B-A4B-it_p300x2_GKAirotr.json",
    "model_id": "id_gemma4-autoport_gemma-4-26B-A4B-it_p300x2",
    "inference_engine": "vLLM",
    "tt_metal_commit": null,
    "vllm_commit": null,
    "model_impl": "gemma4-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/5 passed, 5 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL  | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 4            | 4096 | 128 | 2087.4    | 2087.2        | 2099.9        | 19.7      | 4586.9    | 892.9            | 27.9              | 920.8            | 0.218          |
| 8           | 8            | 4096 | 128 | 24322.0   | 25111.4       | 25112.2       | 188.5     | 48266.4   | 678.9            | 21.2              | 700.1            | 0.166          |
| 16          | 16           | 4096 | 128 | 41223.3   | 41708.1       | 41709.7       | 367.5     | 87895.3   | 745.6            | 23.3              | 768.9            | 0.182          |
| 8           | 8            | 128  | 128 | 3408.2    | 3748.4        | 3749.1        | 180.5     | 26338.0   | 38.9             | 38.9              | 77.8             | 0.304          |
| 16          | 16           | 128  | 128 | 6945.6    | 7280.2        | 7281.1        | 356.6     | 52234.1   | 39.2             | 39.2              | 78.4             | 0.306          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
