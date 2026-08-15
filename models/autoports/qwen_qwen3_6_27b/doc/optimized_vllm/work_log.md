# Qwen3.6-27B optimized vLLM work log

## Frozen comparison contract

Both before/after arms use `Qwen/Qwen3.6-27B`, the selected datatype-sweep
policy, P300x2 physical 1x4 TP4, `FABRIC_1D_RING`, a 200,000,000-byte trace
region, `sample_on_device_mode=all`, explicit temperature 0, and
`max_model_len=262144`.

Primary headline: `max_num_seqs=1`, random 128 input / 128 output, one request,
concurrency 1, ignore EOS. CI burst: `max_num_seqs=32`, random 100 input / 100
output, 32 requests, unconstrained admission, ignore EOS. No context or sampling
contract changed between arms.

## Before and after

| Profile and shape | Metric | Before | After |
|---|---:|---:|---:|
| Primary 128/128/1/c1, maxseq1 | TTFT P50 (ms) | 4138.573 | 3784.303 |
| Primary 128/128/1/c1, maxseq1 | TPOT mean (ms) | 70.733 | 61.893 |
| Primary 128/128/1/c1, maxseq1 | ITL P50 / P99 (ms) | 55.861 / 57.502 | 55.840 / 56.850 |
| Primary 128/128/1/c1, maxseq1 | aggregate output (tok/s) | 9.755 | 10.992 |
| Primary 128/128/1/c1, maxseq1 | TPOT decode (t/s/u) | 14.138 | 16.157 |
| CI 100/100/32, maxseq32 | TTFT P50 / P99 (ms) | 165476.977 / 165478.048 | 162572.518 / 162573.697 |
| CI 100/100/32, maxseq32 | TPOT mean (ms) | 280.063 | 279.381 |
| CI 100/100/32, maxseq32 | ITL P50 / P99 (ms) | 244.015 / 578.093 | 244.131 / 560.379 |
| CI 100/100/32, maxseq32 | aggregate output (tok/s) | 16.785 | 17.049 |

The CI burst's 3.579 t/s/u is capacity/nightly-parity evidence only. The primary
16.157 t/s/u is the headline decode value and is 92.5% of optimized full-model
canonical split-token decode at 17.467 t/s/u.

## Commands

Final full sampling, qualitative, and CI-capacity run:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 \
  --max-num-seqs 32 --max-model-len 262144 --sampling-profile full \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

Final headline reproduction:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 \
  --max-num-seqs 1 --max-model-len 262144 --sampling-profile full \
  --no-benchmark-ci-serving \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

Host contracts: 16 passed. Plugin RNG regression: 1 passed. Full plugin live
sampling: 72 passed, 1 skipped. The combined runner's six-prompt output used raw
`/v1/completions`; independent review correctly rejected it for a model with a
chat template. It is retained only as a diagnostic.

Fresh final qualitative remediation used the exact maxseq1 optimized server:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/vllm_chat_qualitative.py \
  --prompts models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_vllm/artifacts/after_chat \
  --max-tokens 256
python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_vllm/artifacts/after_chat/vllm_chat_qualitative_outputs.json \
  --scope vllm --missing-artifacts critical \
  --json models/autoports/qwen_qwen3_6_27b/doc/optimized_vllm/artifacts/after_chat/degenerate_output_check.json
```

The script sends the same six prompts through `/v1/chat/completions`, preserves
the checkpoint-template rendering and token IDs, and fixes sampled seed
20260815. All 12 outputs were read: coherent, on-topic, grammatical, and free of
wrong-language drift, contamination, or loops. The checker exits 0. Relative to
datatype/full-model and prior vLLM controls, the same healthy reasoning-first
style is preserved. All responses reach the 256-token cap during visible
reasoning, so short-budget instruction completion remains a documented
presentation limitation.

## Optimize checklist and decisions

- Measured the real end-to-end serving bottleneck before editing; redundant
  steady sampler uploads were the isolated avoidable work.
- Preserved selected precision, layouts, sharding, trace shapes, context, mesh,
  and non-aligned support.
- Kept the full-model split sampler and nonblocking trace replay; rejected host
  argmax, full-logits readback, force-argmax, generic eager sampling, aligned-only
  fast paths, and lower context.
- Did not profile this serving stage because the goal prohibits all TT serving
  profilers. Used benchmark JSON and correctness evidence instead.
- The plugin host RNG fix is outside the measured device-sampling path but is
  required for the full sampling gate and async correctness.
- Repeated device-0 core 29-25 heartbeat faults were recovered only with bounded
  reset and mesh smoke. Failed starts were not counted as benchmark evidence.
- Successful final servers shut down cleanly and left no EngineCore process.

## Review and commits

Independent stage review and local commit SHAs are recorded after final review.
Nothing is pushed.
