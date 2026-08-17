## Tenstorrent Model Release Summary: google/gemma-4-26B-A4B-it on P300X2

### Metadata: google/gemma-4-26B-A4B-it on P300X2

```json
{
    "model_name": "google/gemma-4-26B-A4B-it",
    "device": "P300X2",
    "generated_at": "2026-08-17T14:25:03+00:00",
    "report_id": "id_autoport-Gemma-4-26B-A4B-it_p300x2_releaseflow_2026-08-17_14-27-02",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --workflow release --runtime-model-spec-json /home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/autoport_releaseflow_spec.json --tt-device p300x2 --service-port 8000 --server-url http://127.0.0.1 --no-auth --limit-samples-mode smoke-test",
    "runtime_model_spec_json": "/tmp/claude-2896121/-home-mvasiljevic-tt-metal/8648d126-70e2-4f72-8074-38818c035e90/scratchpad/tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-17_14-08-34_id_autoport-Gemma-4-26B-A4B-it_p300x2_releaseflow_dVN1oxoU.json",
    "model_id": "id_autoport-Gemma-4-26B-A4B-it_p300x2_releaseflow",
    "model_repo": "google/gemma-4-26B-A4B-it",
    "inference_engine": "vLLM",
    "tt_metal_commit": "4b17e185dea9e70db276bcd5b0ed47c5786738b0",
    "vllm_commit": "938c45ed71f3f669ffd38e4c9a033c3391cec961",
    "model_impl": "autoport-gemma4-26b-a4b-it"
}
```

### Acceptance Criteria

- Acceptance status: ❌ `FAIL`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (1/1 passed)
- Evals: ❌ `FAIL` (0/3 passed, 2 failed, 1 NA)
- Spec Tests: 🟨 `NA` (no blocks present)

#### Blockers

- `evals:LLM Eval — terminal_bench_2`: LLM Eval — terminal_bench_2 reported success=False (attempts=?)
- `evals:LLM Eval — swe_bench_verified`: LLM Eval — swe_bench_verified reported success=False (attempts=?)

---

### Accuracy Evaluations for google/gemma-4-26B-A4B-it on P300X2

| Task               | Tolerance | Published Score Ref                     | gpu_reference_score_ref | Score | Ratio to Published | Ratio to Reference | Accuracy Check | Success | subprocess_rc |
|:-------------------|:----------|:----------------------------------------|:------------------------|:------|:-------------------|:-------------------|:---------------|:--------|:--------------|
| r1_gpqa_diamond    | 0.05      | https://huggingface.co/Qwen/Qwen3.6-27B | TBD                     | 100   | N/A                | N/A                | 🟨 NA          | N/A     | N/A           |
| terminal_bench_2   | 0.05      | https://huggingface.co/Qwen/Qwen3.6-27B | N/A                     | N/A   | N/A                | N/A                | ❌ FAIL        | false   | 1             |
| swe_bench_verified | 0.05      | https://huggingface.co/Qwen/Qwen3.6-27B | N/A                     | N/A   | N/A                | N/A                | ❌ FAIL        | false   | 1             |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for google/gemma-4-26B-A4B-it on P300X2

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 261.2     | 261.5         | 265.0         | 35.6      | 4787.5    | 26.7              | 0.209          | ✅ PASS      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 1000        | 0.2612     | ✅ PASS    | 10.00            | 2.806           | ✅ PASS         | 10                 | 2.674             | ✅ PASS           |
| complete   | 500         | 0.5224     | ✅ PASS    | 20.00            | 1.403           | ✅ PASS         | 20                 | 1.337             | ✅ PASS           |
| target     | 300         | 0.8706     | ✅ PASS    | 26.00            | 1.079           | ✅ PASS         | 26                 | 1.028             | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.
