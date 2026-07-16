# Gemma 4 31B Optimized vLLM Serving

## Headline

Primary single-user TT plugin serving improved to **494.656 ms TTFT** and
**26.645 decode t/s/u** for 127 actual input tokens (128 requested), 128 output
tokens, one request, concurrency 1, temperature 0, and ignore-EOS. This is the
real vLLM path through `tt/generator_vllm.py`, on a `1x4` P150b mesh with
`max_num_seqs=32`, `max_model_len=113280`, `sample_on_device_mode=all`, and
async scheduling.

| Same primary workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Aggregate output throughput | TPOT decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before: 127 actual input (128 requested), 128 output, 1 request, concurrency 1 | 992.586 / 992.586 ms | 38.023 / 38.023 ms | 29.348 / 29.739 ms | 21.974 tok/s | 26.300 |
| After: 127 actual input (128 requested), 128 output, 1 request, concurrency 1 | 494.656 / 494.656 ms | 37.531 / 37.531 ms | 29.330 / 32.840 ms | 24.328 tok/s | 26.645 |

Relative to the same-runner baseline, mean TTFT fell 50.16%, mean TPOT fell
1.29%, aggregate output throughput rose 10.71%, and TPOT-derived decode rose
1.31%. The before/after benchmark JSON is retained under `before/` and
`after/`.

## Secondary CI serving burst

This is capacity and nightly-parity evidence, not headline decode t/s/u.
Burst admission and interleaved prefill affect TPOT.

| Same CI burst workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Aggregate output throughput | TPOT-derived burst rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before: 99 actual input/request (100 requested), 100 output/request, 32 requests, burst concurrency up to 32 | 8485.248 / 8488.457 ms | 77.373 / 127.442 ms | 55.807 / 687.715 ms | 201.070 tok/s | 12.924 t/s/u |
| After: 99 actual input/request (100 requested), 100 output/request, 32 requests, burst concurrency up to 32 | 7956.575 / 7960.168 ms | 75.800 / 125.780 ms | 55.831 / 631.891 ms | 210.102 tok/s | 13.193 t/s/u |

## Optimization and async contract

Stage 09 already used the selected `lm_head_bfp8_hifi2` policy, split greedy
sampling, persistent trace tensors, and nonblocking model/sampler replay. The
remaining steady-token adapter overhead was scheduler input inspection: stale
host token and position tensors were parsed and all hybrid page tables were
compared even when vLLM knew the batch and tables were unchanged.

The TT async controller now passes `reuse_device_decode_inputs=True` only when
its existing `can_use_steady_decode_fast_path()` invariants hold. Under that
explicit handshake, `decode_forward(..., read_from_device=False)` returns the
persistent device token tensor by calling `decode_next_token_traced()` directly:

- the prior device sampler wrote the next token into persistent
  `trace_state.token_input`;
- the model trace advanced persistent RoPE/cache position tensors;
- persistent model and sampler inputs stay device-resident;
- unchanged scheduler page tables are not recopied or compared;
- model and sampler traces use `ttnn.execute_trace(..., blocking=False)`;
- the plugin schedules the deferred read only after the submission boundary.

Any reset, batch-size change, cache allocation change, sampling-key change, or
unproven page-table state takes the validated slow control path. That path
releases live traces before allocation, refreshes changed token, position, and
page-table inputs, and recaptures only when required. Contract tests cover
changed token/current-position state, changed page tables, unchanged page
tables, reset, dynamic active prefixes, and stale host tensors on the proven
steady path.

KV block allocation is tracked separately from `reset_batch`: scheduler
`new_block_ids` disables pending-step overlap and steady reuse for exactly the
boundary step. The existing adapter path refreshes the persistent page-table
buffers and generations without trace recapture, then the next unchanged token
resumes direct replay. This matters for the 64-token block size and is covered
by focused host tests. The exact persistent page-table copy program is warmed
before trace registration; this prevents first-boundary program allocation
while traces are live without changing scheduler-state semantics.

`supports_async_decode=True` and `supports_async_decode_overlap=True` are
exercised by the final server with `--async-scheduling`; the server log records
`TTScheduler, async_scheduling=True` and successful serving through both
benchmarks. The optimized branch is not an aligned-input specialization: the
same run passed a 149-token prompt, whose length is nonzero modulo 32, 64, and
128.

## Sampling and full-model comparison

Both measured benchmarks use temperature 0 and `sample_on_device_mode=all`.
They reuse the full-model greedy TP4 split sampler. There is no host argmax,
full-logits readback, eager generic sampler, force-argmax path, or host
token-feedback loop in the measured decode path. The explicit
`GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1` switch exists only so the shared full
sampling test suite can exercise stochastic and structured-output features
that are outside the model-scoped greedy device policy.

Comparable full-model evidence is:

- selected traced teacher forcing: 24.561 t/s/u;
- selected standalone token-out end-to-end decode: 24.787 t/s/u;
- selected standalone steady token-out decode: 34.256 t/s/u;
- optimized vLLM TPOT decode: 26.645 t/s/u;
- optimized vLLM median ITL rate: 34.095 t/s/u.

The 24.561 t/s/u teacher-forcing result is a useful traced lower-bound/ranking
control, not a split-sampling token-out measurement. Against the directly
relevant token-out control, vLLM TPOT is 7.49% above end-to-end decode. Its
median ITL rate is within 0.47% of standalone
steady token-out, so there is no material serving-specific steady-decode gap
for comparable, though not identical, work. The exact standalone source is
`../datatype_sweep/post_selection_token_out.json` (prompt 149, generated 100,
batch 1, 99 steady replays, one sampled-token read, no full-logits readback).

## Correctness, context, and quality gates

- Adapter and full-model trace contracts: 49 passed; JUnit
  `evidence/adapter_contract.xml`.
- Plugin lane/scheduler contracts: 22 passed; JUnit
  `evidence/plugin_lane_contract.xml`.
- Full vLLM sampling: 72 passed, 1 skipped; `after/sampling_tests.log`.
- Non-aligned serving: 149 input + 1 output, HTTP 200;
  `after/non_aligned_prompt_check.json`.
- Context ceiling: 113279 input + 1 output, HTTP 200;
  `after/max_context_prompt_check.json`.
- Logit stability: chosen token 108 and top-20 logprobs exactly equal across
  repeated runs and batch positions 0/1/2; `after/logit_determinism.json`.
- Qualitative: six greedy and six sampled raw continuations reviewed; zero
  mechanical degeneracy findings, exit 0; see `qualitative/verdict.md`.

The context contract is unchanged. vLLM remains at the proven physical ceiling
of 113280 tokens on four P150b devices; the standalone/full-model contract
remains 262144. No benchmark, evaluation, or qualitative context was lowered.

## Reproduction command

The before and after primary/CI measurements use this identical runner,
generation mode, mesh, max sequence count, maximum length, sampling mode, and
TT config:

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

## Limitations and hygiene

- The 113280 vLLM limit is the documented hard HMA cache/live-set physical
  ceiling; 113344 is short 148800 bytes per bank. This stage did not reduce it.
- The base checkpoint is a raw continuation model and is weak at instruction
  following; this matches the selected standalone control rather than a serving
  regression.
- Tracy, `tt-perf-report`, device profiling, adapter profiling, and
  `ReadDeviceProfiler` were intentionally not used for this serving stage.
- Two device-open attempts during this stage hit the same device-0 Ethernet
  resume failure before model code. Each used the bounded list/reset/list
  recovery and a passing `1x4` mesh smoke before resuming. The final runner
  shut down cleanly; all four
  devices list healthy and there is no live vLLM, API-server, runner, or
  EngineCore process. Historical PID-1-owned zombies are not device holders.
- The final fresh server log has zero unsafe allocator warnings, tracebacks, or
  error matches. Exact sampler prewarm precedes model/sampler capture, and the
  persistent page-table copy is prewarmed before any trace registration.
  Nanobind reference-leak diagnostics remain at interpreter shutdown; requests,
  device close, and process cleanup all completed.
- Firmware 19.9 is newer than the latest fully tested 19.5 bundle.

See `work_log.md`, `runtime_fallback_audit.md`, and `perf_summary.json` for the
full decision and artifact record.

Independent Stage 10 review: `clean-pass`; see `stage_review.md`.
