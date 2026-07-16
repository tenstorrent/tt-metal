# Gemma 4 31B vLLM Integration

## Headline

Primary single-user vLLM TT serving, final traced on-device greedy path:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | Decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 127 actual input tokens (128 requested), 128 output tokens, 1 request, concurrency 1, temperature 0.0, ignore EOS | 992.586 / 992.586 ms | 38.023 / 38.023 ms | 29.348 / 29.739 ms | 21.974 output tok/s | 26.300 t/s/u |

CI serving-burst evidence, secondary only:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | TPOT-derived t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 99 actual input tokens/request (100 requested), 100 output tokens/request, 32 requests, concurrency up to 32 under burst admission, temperature 0.0, ignore EOS | 8485.248 / 8488.457 ms | 77.373 / 127.442 ms | 55.807 / 687.715 ms | 201.070 output tok/s | 12.924 t/s/u |

The CI burst profile is not the headline decode rate: burst admission and
prefill scheduling can affect TPOT. The primary single-user decode result is
`1000 / mean_tpot_ms` from the 127-actual-input/128-output workload.

## Serving Status

- Model: `google/gemma-4-31B`
- Hardware: four Blackhole P150b devices, `1x4` mesh, tensor parallel degree 4
- Adapter: `models/autoports/google_gemma_4_31b/tt/generator_vllm.py`
- Plugin selector: `TT_GEMMA4_TEXT_VER=gemma4_31b_autoport`
- Served max model length: `113280`, matching
  `doc/context_contract.json::vllm_supported_context`
- Standalone/full-model context: `262144`; the vLLM-only reduction is a hard
  physical HMA-cache-plus-prefill-live-set limit
- Max sequences: `32`; vLLM KV block size: `64`
- Sampling profile: `full`; the optional shared-test host path is explicitly
  enabled with `GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1`
- Qualitative prompt format: raw base-model continuation through
  `--qualitative-raw-prompts`; the tokenizer has no chat template
- TT config: `{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}`
- Async decode: `supports_async_decode=True`; server uses `--async-scheduling`
- Decode trace: enabled, using the canonical full-model split model/sampler
  trace and token-out feedback path

The adapter constructs the selected `lm_head_bfp8_hifi2` precision policy:
BFP8 attention weights, BFP4 MLP weights, BFP8 LM head, BF16 activations and
residuals, BF16 prefill CCL, BFP8 decode CCL, BFP8 KV cache, BF16 logits,
LoFi attention/MLP compute, HiFi2 LM-head compute, FP32 sampling-gather values,
and no layer exceptions.

Startup may read temporary demo-constructor BFP8 MLP cache files before each
layer replaces that object with the TP-optimized shared BFP4 MLP. The selected
physical runtime policy is recorded in
`doc/datatype_sweep/post_selection_token_out.json::runtime_precision`.

vLLM owns the serving KV cache and page tables. The adapter disables standalone
cache allocation, requires external cache handles, and passes the hybrid page
tables into the full-model generator. Greedy serving delegates to
`prepare_token_out_decode()` and `decode_next_token_traced()`. The measured path
has no separate sampler, host argmax, full-logits readback, generic top-k
fallback, or Python token readback/writeback feedback loop.

## Context Capacity Evidence

Gemma 4 has fifty sliding-attention and ten global-attention layers. The vLLM
hybrid cache groups share ten physical K/V workspaces across six logical
attention groups. Full `262144` serving (`22533` pool blocks) failed full-depth
HMA KV allocation; `157696` (`13557` blocks) failed on physical KV buffer 19 of
20.

The final source model uses:

- physical pool blocks `B(C) = 5 * (ceil(C / 64) + 1) + ceil(C / 128)`;
- post-KV largest contiguous bytes/bank
  `P(C) = 2178911936 - 174080 * B(C)`;
- sixty physically aligned page tables
  `T(C) = 60 * 4 * ceil((4 * ceil(C / 64)) / 32) * 32` bytes/bank;
- fused streamed-attention peak `A(C) = 4096 * C + 4032 * 4096` bytes/bank.

At `C=113280`, the pool is `9740` blocks, `P=483372736` bytes/bank,
`A=480509952` bytes/bank, page tables consume `1704960` bytes/bank, and the
remaining runtime margin is `1157824` bytes/bank. The adjacent aligned
candidate `113344` needs `9746` blocks; its `482328256` post-KV bytes/bank are
`148800` bytes/bank short of its `480772096` mandatory peak plus the same
`1704960` page tables. Thus `113344` is source-proven physically infeasible.

The capacity fixes stream both attention-output projection/post-attention
residual work and the long-prompt MLP residual path, release dead prompt-sized
normalization lifetimes, and use a zero-copy global-attention cache read view
with layer geometry. The direct final gate returned a real completion for
`113279` input tokens plus one output token. The gate rejects HTTP-200 error
envelopes, so startup or a short request is not accepted as capacity proof.
See `context_capacity_audit.md` and
`evidence/full_113280_source_ceiling_max_context_passing_server.log`.

## Final Command

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHONPATH=$PWD/vllm:$PWD:$PWD/ttnn:$PWD/tools \
LD_LIBRARY_PATH=$PWD/build/lib:/opt/openmpi-v5.0.7-ulfm/lib \
TT_GEMMA4_TEXT_VER=gemma4_31b_autoport \
GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1 \
GEMMA4_31B_TENSOR_CACHE=/tmp/gemma4_31b_full_model_tensor_cache \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python -u -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --stages serve,sampling,qualitative,benchmark \
  --sampling-profile full \
  --qualitative-raw-prompts \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 113280 \
  --check-max-context-prompt \
  --block-size 64 \
  --server-timeout 2400 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}' \
  --additional-server-args='--async-scheduling --chat-template /localdev/odjuricic/tt-metal/models/autoports/google_gemma_4_31b/doc/vllm_integration/chat_template.jinja'
```

The runner also submits a direct 149-token-ID completion request. Length 149 is
not divisible by the 64-token page, 32-row tile, or 128-token trace size.

The server command launched by that runner was:

```bash
/opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --block_size 64 --max_num_seqs 32 --port 8000 \
  --max_model_len 113280 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}}' \
  --async-scheduling \
  --chat-template /localdev/odjuricic/tt-metal/models/autoports/google_gemma_4_31b/doc/vllm_integration/chat_template.jinja
```

## Validation and Artifacts

- `readiness_vllm/non_aligned_prompt_check.json`: 149 input + 1 output pass
- `readiness_vllm/max_context_prompt_check.json`: 113279 input + 1 output pass
- `readiness_vllm/logit_determinism.json`: exact repeated and batch-position
  top-20 logits, with the standalone-TT greedy winner comparison
- `readiness_vllm/sampling_tests.log`: `72 passed, 1 skipped`
- `readiness_vllm/vllm_qualitative_outputs.json`: all twelve raw continuations
- `readiness_vllm/degenerate_output_check.json`: zero findings, exit 0
- `readiness_vllm/qualitative_verdict.md`: human per-prompt judgment
- `readiness_vllm/vllm_result.json` and `vllm_benchmark.json`: raw and
  normalized primary benchmark
- `readiness_vllm/vllm_ci_serving_result.json` and
  `vllm_ci_serving_benchmark.json`: raw and normalized secondary CI burst
- `readiness_vllm/vllm_benchmark.log`, `vllm_ci_serving_benchmark.log`, and
  `server.log`: benchmark and shared-server logs
- `doc/vllm_integration/evidence/final_full_113280_vllm_run.log`: final runner
  transcript
- `doc/vllm_integration/evidence/full_113280_source_ceiling_max_context_passing_server.log`:
  independent source-ceiling capacity pass
- `doc/vllm_integration/evidence/context_capacity_audit.json`: machine-readable
  final capacity derivation; `context_capacity_audit.md` is its narrative form

## Qualitative Verdict

Pass for serving integrity, not for strong instruction following. All twelve
greedy/sampled raw continuations were read. They are grammatical and remain in
the expected language, with no gibberish, wrong-language drift, cross-request
state corruption, or stale-token feedback. The mechanical checker exits zero.

The base model often continues an apparent request corpus instead of answering:
both supervised-learning trajectories become question lists, both translation
trajectories list new translation exercises, both thermodynamics trajectories
enumerate related questions, and the story continuations quickly become lists
of writing requests. Greedy supervised learning repeats one question and the
greedy story loops two writing prompts. The haiku pair begins on topic before
continuing related prompts; both Fibonacci outputs are useful and on topic.
These weaknesses match Stage 08 raw-continuation controls. See
`readiness_vllm/qualitative_verdict.md`.

## Performance Comparison

The datatype-sweep teacher-forcing result, `24.561 t/s/u`, is a lower bound,
not the serving result. Stage 08 standalone token-out evidence measured
`479.707 ms` TTFT, `24.787 t/s/u` end-to-end decode, and `34.256 t/s/u` steady
decode. Final vLLM measures `992.586 ms` TTFT and `26.300 t/s/u`; ITL P50/P99
is `29.348/29.739 ms`. The result exceeds the teacher-forcing lower bound and
leaves no evidenced avoidable vLLM-specific steady-decode overhead in the
measured path.

## Runtime, Cleanup, and Limitations

- Prefix caching is disabled.
- Full-depth HMA vLLM serving is physically limited to 113280 tokens on the
  tested four-P150b system; standalone capability remains 262144.
- On-device sampling is greedy-only. Shared stochastic, penalty, allowlist,
  structured-output, and logprob tests may use the explicit optional host path;
  neither benchmark uses it.
- The final runner terminated API server and EngineCore cleanly. Post-run audit
  found no live device holders; only historical PID-1 zombie records remained.
- Firmware bundle 19.9 is newer than the latest fully tested 19.5 bundle.
- CI burst metrics are secondary and are not used as headline decode t/s/u.
