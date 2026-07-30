# Optimized Full Model Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.2-1B-Instruct`

Mesh: T3K `1x8`, `FABRIC_1D_RING`

## Code Changes

- Added optional `num_layers` construction to `Llama32FullModel` and
  `Llama32Generator` for reduced real-weight profiler runs.
- Enabled `pad_logits_to_power_of_2=True` for sampling.
- Updated `SplitGreedySampler` to pad the local 16032-column logits shard to
  16384 before local top-k, preserving semantic greedy selection.
- Added `benchmark_token_out_no_readback()` for traced full-model token-out
  measurement with persistent token feedback and no per-token host boundary.
- Added optimized-full-model tests for no-readback benchmark, reduced profiler
  signposts, reduced eager full-path profiling, and guarded runtime fallback.

## Commands And Results

Syntax:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py
```

Initial no-readback token-out benchmark, 128-token synthetic prompt:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
MD_OPT_FULL_MODEL_PROMPT_LEN=128 MD_OPT_FULL_MODEL_DECODE_STEPS=128 \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_token_out_no_readback_benchmark
```

Result: passed, 6.2518 ms/token, 159.95 t/s/u.

Comparable final no-readback token-out benchmark, 60-token prompt:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
MD_OPT_FULL_MODEL_PROMPT_LEN=60 MD_OPT_FULL_MODEL_DECODE_STEPS=128 \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_token_out_no_readback_benchmark
```

Result: passed, TTFT 179.9986 ms, 6.2513 ms/token, 159.9662 t/s/u.
Measured-loop counters: 128 model trace replays, 128 sampling trace
replays/calls, zero token/position/RoPE/page-table refreshes, zero readbacks,
one final sync.

Runtime fallback audit:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_runtime_fallback_audit_no_readback_loop
```

Result: passed. The guarded loop rejects `ttnn.from_torch`, `ttnn.to_torch`,
`ttnn.copy_host_to_device_tensor`, and `ttnn.synchronize_device`; all guarded
bridges remained unused during two measured replays.

Reduced traced profiler attempt:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
MD_OPT_FULL_MODEL_PROMPT_LEN=128 MD_OPT_FULL_MODEL_REDUCED_DECODE_STEPS=8 MD_OPT_FULL_MODEL_REDUCED_LAYERS=1 \
python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model/tracy/reduced_token_out \
  -m pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_reduced_perf_artifact_signposts
```

Result: hung after the decode signpost and emitted no `ops_perf_results` CSV.
The process was stopped; `tt-smi -r all` reset the local boards. Hardware came
back healthy.

Reduced eager full-path profiler:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
MD_OPT_FULL_MODEL_PROMPT_LEN=128 MD_OPT_FULL_MODEL_REDUCED_LAYERS=1 \
python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model/tracy/reduced_eager_full_path \
  -m pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_reduced_eager_profile_signposts
```

Result: passed. Raw artifacts were copied to `perf/`, then `tt-perf-report`
generated merged and per-device reports for `PERF_FULL_MODEL_PREFILL` and
`PERF_FULL_MODEL_EAGER_DECODE`.

AIME24 prefill:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 88/100, top5 100/100, top100 100/100.

AIME24 teacher forcing:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt \
  --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 86/100, top5 100/100, top100 100/100, TTFT 277.31 ms,
decode 91.89 t/s/u, e2e 73.47 t/s/u.

Autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --mesh-device T3K --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-dir models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model/artifacts/autoregressive_default_128
```

Result: HF produced 128 tokens, TT produced 128 tokens.

Degenerate-output audit:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --root models/autoports --scope autoregressive --missing-artifacts critical
```

Result: no degenerate output detected. Optimized TT completion adjacent
duplication 0.0, trigram loop fraction 0.0841.

Top-k/top-p smoke:

```bash
python - <<'PY'
# Instantiates Llama32Generator on T3K/FABRIC_1D_RING and calls:
# generate(..., max_new_tokens=16, top_k=16, top_p=0.9, temperature=0.8)
PY
```

Result: passed; `artifacts/trace_evidence/topk_topp_trace_smoke.json` reports
one common sampling internal trace and force-argmax disabled.

Watcher:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_runtime_fallback_audit_no_readback_loop
```

Result: passed in 691.97 s. Watcher summary has zero critical issue matches.

## Perf Report Notes

`tt-perf-report` reduced eager decode summary:

- Matmuls: 730.78 us, 47.60%.
- TopKDeviceOperation: 154.17 us, 10.04%.
- Candidate all-gather for split sampler: 63.13 us.
- Sampler argmax/gather/pack: 166.02 us.
- LM-head matmuls: 619.60 us.

The largest terminal cost is LM head, not force-argmax, full-vocab all-gather,
or generic sampler overhead. Full traced token-out is not slower than the
decoder stack lower bound derived from isolated layer timing, so no additional
gap split was required.
