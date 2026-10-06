## Tenstorrent Model Release Summary: google/gemma-4-26B-A4B-it on P300X2

### Metadata: google/gemma-4-26B-A4B-it on P300X2

```json
{
    "model_name": "gemma-4-26B-A4B-it",
    "model_repo": "google/gemma-4-26B-A4B-it",
    "device": "P300X2",
    "generated_at": "2026-09-29T13:23:35+00:00",
    "report_id": "google__gemma-4-26B-A4B-it_2026-09-29T132335+0000",
    "workflow": "benchmarks",
    "report_partial": false,
    "report_blocks": 1,
    "server_mode": "docker",
    "run_command": "python run.py --model google/gemma-4-26B-A4B-it --workflow benchmarks --device p300x2 --docker-server --dev-mode --impl gemma4-autoport --ci-mode --override-docker-image ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.22.0-eb1d2af61c7630c7424e9593b646093ef7be45af-c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42-ttmetal-7f72b1c6e905-109393485980",
    "runtime_model_spec_json": "/home/ubuntu/actions-runner/_work/tt-agentic-bringup-qb2/tt-agentic-bringup-qb2/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-29_13-05-07_id_gemma4-autoport_gemma-4-26B-A4B-it_p300x2_3q-5FQ5g.json",
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
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- Agentic Targets: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL  | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 1           | 4            | 4096 | 128 | 2090.0    | 2092.1        | 2100.9        | 19.7      | 4596.0    | 891.2            | 27.8              | 919.0            | 0.218          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
