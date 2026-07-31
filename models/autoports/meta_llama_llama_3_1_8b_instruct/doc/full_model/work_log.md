# Full Model Work Log

Date: 2026-06-15. Branch:
`agentic-research/experiment-17-llama31-8b`, commit `86f8bc022e6`.

## Files

- `tt/model.py`
- `tt/generator.py`
- `tt/multichip_decoder.py`
- `models/common/readiness_check/generate.py`
- `doc/full_model/README.md`
- `doc/full_model/work_log.md`
- `doc/full_model/token_out_trace_evidence.py`
- `doc/full_model/reduced_profile.py`

## Implementation Notes

The full model wraps the optimized multichip decoder stack with embedding,
RoPE, final RMSNorm, split LM head, paged KV-cache ownership, and a readiness
generator. Decode captures a token-input-to-logits model trace and uses
`SamplingGenerator` internal trace for split sampling. The sampler writes back
to the persistent decode token input through `tt_out_tok`.

Fixes made during bring-up:

- Added `BatchEncoding`/dict handling for tokenizer chat-template output in
  `models/common/readiness_check/generate.py`.
- Fixed long-prefill multichip MLP `out_subblock_w` selection so prefill no
  longer hits invalid matmul config for gate/up.
- Added a persistent decode embedding all-gather output buffer before first
  capture so trace replay does not miss the program cache.
- Fixed generator reset handling for reused traces by treating an unknown
  `_decode_host_position` as a position mismatch and resetting persistent
  token/position/RoPE/page-table inputs once.

## Commands And Results

Compile:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  models/common/readiness_check/generate.py
```

Result: passed.

Fresh AIME24 reference:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.generate \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt
```

Result: wrote one `readiness_v1` entry with prompt length 184, generated length
100, top-k 100.

Prefill readiness:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `0.900 (90/100)`, top5 `1.000 (100/100)`,
top100 `1.000 (100/100)`.

Teacher forcing:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 `0.920 (92/100)`, top5 `1.000 (100/100)`,
top100 `1.000 (100/100)`, TTFT `1094.60 ms`, decode `22.18 t/s/u`,
e2e `17.99 t/s/u`.

Autoregressive readiness:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/autoregressive_story_100
```

Result: HF and TT each produced 100 tokens. HF begins with "a small, shimmering
light hovering above the grass"; TT begins with the same phrase and continues as
a coherent English story about the light/crystal. No repetition, wrong-language
drift, or early feedback collapse was observed.

Degeneracy:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --missing-artifacts critical \
  --scope autoregressive \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/autoregressive_degenerate_report.json
```

Result: no findings.

Token-out trace evidence:

```bash
python_env/bin/python -u \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.json \
  > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence_stdout.txt
```

Result artifact: `token_out_trace_evidence.json`. Summary:
TTFT `616.64 ms`, decode including first trace capture `49.65 t/s/u`, steady
replay decode `69.21 t/s/u`, trace probe passed, top-k/top-p smoke passed.

Reduced profiler:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy/reduced_profile/.logs
python_env/bin/python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy/reduced_profile/.logs \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile_summary.json
```

Result: passed. First run compiled profiler-instrumented kernels and wrote
`ops_perf_results_2026_06_15_17_09_31.csv`.

Stable report generation:

```bash
BASE=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/tracy/reduced_profile
python_env/bin/tt-perf-report "$BASE/reduced_profile_ops_perf_results.csv" \
  --start-signpost PERF_REDUCED_PREFILL \
  --end-signpost PERF_REDUCED_PREFILL_END \
  --csv "$BASE/reduced_prefill_perf_report.csv" \
  > "$BASE/reduced_prefill_perf_report.txt"
python_env/bin/tt-perf-report "$BASE/reduced_profile_ops_perf_results.csv" \
  --start-signpost PERF_REDUCED_TOKEN_OUT_DECODE \
  --end-signpost PERF_REDUCED_TOKEN_OUT_DECODE_END \
  --csv "$BASE/reduced_token_out_decode_perf_report.csv" \
  > "$BASE/reduced_token_out_decode_perf_report.txt"
python_env/bin/tt-perf-report "$BASE/reduced_profile_ops_perf_results.csv" \
  --start-signpost PERF_REDUCED_TOKEN_OUT_DECODE \
  --end-signpost PERF_REDUCED_TOKEN_OUT_DECODE_END \
  --no-merge-devices \
  --csv "$BASE/reduced_token_out_decode_perf_report_per_device.csv" \
  > "$BASE/reduced_token_out_decode_perf_report_per_device.txt"
```

Result: reduced prefill 42 merged device ops and 1643.152 us device time;
reduced token-out decode 61 merged device ops and 1368.926 us device time.

## Runtime Fallback Audit

Clean for the full-model stage:

- no single-chip model fallback;
- no per-token host argmax in decode;
- no full-vocab logits readback in decode;
- no untraced model decode in readiness teacher forcing or token-out evidence;
- no Python token feedback loop; sampled token is written to persistent decode
  input on device;
- no per-token host rebuild/copy of token, current position, RoPE index, masks,
  or unchanged page table in steady free-running decode;
- changed page table copies are detected and performed once;
- KV caches are model-owned and reused.

## Limitations

- First sampling trace capture currently emits an allocator warning because the
  sampler precompile/capture happens after model trace capture. The traced
  split-sampling path succeeds and evidence proves feedback and state coherence.
- Teacher-forcing readiness is slower than free-running token-out steady replay
  because it feeds the reference token from the host each step. The docs report
  both paths separately.
- Reduced `tt-perf-report` op-to-op gap sums are larger than the measured host
  replay window, so they are treated as profiler/signpost context rather than
  evidence of a host-stepped decode loop. Trace counters are the fallback audit
  source of truth for per-token host work.
