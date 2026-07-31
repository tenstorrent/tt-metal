# Optimized vLLM Status

Status: **complete** on 2026-06-15 for `microsoft/Phi-3.5-mini-instruct`.

Top batch-1 vLLM TT serving result: prefill TTFT P50/P99/mean **173.13 / 284.60 / 177.20 ms**, decode **51.44 t/s/u mean per user**, ITL P50/P99/mean **19.39 / 20.61 / 19.44 ms**, aggregate output throughput **48.33 tok/s**. Workload: prompt_len=128, output_len=128, 32 requests, concurrency=1, `--max-num-seqs 1`, `--max-model-len 512`, `--block-size 32`, T3K, `sample_on_device_mode=all`, `trace_region_size=200000000`, `FABRIC_1D_RING`.

The measured path is real vLLM TT plugin serving through `tt/generator_vllm.py`; the final run launched `python -m vllm.entrypoints.openai.api_server` via `python -m models.common.readiness_check.run_vllm_server` and served through the TT plugin config.

## Serving Metrics

Before artifact: `readiness_vllm/vllm_benchmark_before_optimized_vllm.json`.
After artifact: `readiness_vllm/vllm_benchmark.json`.

| Metric | Before | After |
| --- | ---: | ---: |
| Completed requests | 32 | 32 |
| Prompt/output length | 128 / 128 | 128 / 128 |
| Concurrency | 1 | 1 |
| TTFT P50 / P99 / mean | 170.79 / 222.99 / 173.47 ms | 173.13 / 284.60 / 177.20 ms |
| ITL P50 / P99 / mean | 19.33 / 21.11 / 19.45 ms | 19.39 / 20.61 / 19.44 ms |
| Aggregate output throughput | 48.38 tok/s | 48.33 tok/s |
| Mean per-user decode | 51.41 t/s/u | 51.44 t/s/u |

The after run is effectively throughput-neutral relative to the vLLM integration baseline while satisfying the optimized serving contract: async split-read decode, nonblocking token-out trace replay, on-device sampling, and changed-only scheduler input refresh.

Optimized full-model comparison:

- selected datatype token-out no-readback: **56.37 t/s/u** (`doc/datatype_sweep/perf/post_selection_token_out_no_readback_prompt128_gen128.json`)
- optimized full-model token-out no-readback: **56.43 t/s/u** (`doc/optimized_full_model/perf/token_out_no_readback_prompt128_gen128_lmhead8192.json`)
- optimized vLLM serving decode: **51.44 t/s/u**, about **91.2%** of optimized full-model token-out for comparable prompt/output work.

## Correctness

Final `run_vllm_server` command:

```bash
VLLM_TT_SKIP_HOST_ONLY_SAMPLING_TESTS=1 timeout 3600 python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --block-size 32 \
  --sampling-profile full \
  --server-timeout 1800 \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

Results:

- vLLM TT plugin sampling tests: **59 passed, 13 skipped** in `readiness_vllm/sampling_tests.log`.
- Qualitative outputs: coherent and on-topic greedy/sampled completions in `readiness_vllm/vllm_qualitative_outputs.json`.
- Degenerate-output scan: **No degenerate output detected**.
- Serving benchmark: **pass**, `readiness_vllm/vllm_benchmark.json`.
- Contract checks: `readiness_vllm/optimized_vllm_contract_checks.json` and `readiness_vllm/adapter_contract_checks.json` both `pass`.

## Trace And Async Contract

The adapter declares `supports_async_decode=True`, `supports_sample_on_device=True`, `supports_prefix_caching=False`, and `tt_async_decode_allows_overlap=False`. Overlap remains disabled because vLLM still builds the next request state from host scheduler inputs.

`decode_forward(..., read_from_device=False)` delegates to `Phi35MiniGenerator.decode_forward_token_out(...)` and returns the device sampled token output. `read_decode_output(..., async_read=True)` performs the deferred nonblocking CPU read and records the event; `process_decode_output_host(...)` only formats host token/logprob tensors.

vLLM owns the attention KV cache in serving mode. The adapter initializes the generator with `allocate_standalone_cache=False`, allocates the vLLM KV cache tensors, and passes `kv_cache`, `page_table`, token, and current-position inputs by identity into the generator; it does not substitute a standalone cache.

Serving decode captures a token-out trace with `capture_sampling=True`: model decode, LM head, `SamplingGenerator.sample(...)`, and `tt_out_tok=self.trace.token_input` are inside the serving trace. Hot replay calls `ttnn.execute_trace(..., blocking=False)` and returns `self.trace.sampled`. Standalone optimized full-model generation still keeps the split-sampling trace path.

Persistent device inputs are used for token, current position/RoPE state, page table, KV cache, and sampler token feedback. Token and current-position inputs are refreshed only on reset/capture/scheduler mismatch; page tables are compared on host and unchanged page tables return the existing persistent tensor without a per-token copy. Changed page tables update the persistent tensor in place.

Stale-input coverage is split across `adapter_contract_checks.json`, `optimized_vllm_contract_checks.json`, and the trace smoke artifact: adapter checks verify token/current-position/page-table/KV identity delegation, optimized checks verify changed-page-table in-place copy and unchanged-page-table no-copy branches, and `optimized_vllm_trace_variant_smoke.json` verifies nonblocking no-readback replay with persistent token feedback.

## Runtime Audit

The final server log contains no `Traceback`, runtime `ERROR`, host greedy/argmax fallback, force-argmax path, full-logits readback, eager sampling fallback, `TopKDeviceOperation`, `ArgMaxDeviceOperation`, `blocking=True` trace replay, `ReadDeviceProfiler`, `TT_METAL_DEVICE_PROFILER`, `tt-perf-report`, or `tracy` signature. The only `fallback` matches are TTNN/vLLM config text.

There is one allocator warning in `readiness_vllm/server.log`:

```text
Allocating device buffers is unsafe due to the existence of an active trace.
```

This happens at a request-boundary prefill while the persistent decode trace is live, not in the hot decode replay. Prefill token/logit/sample temporaries are explicitly deallocated after host readback. Releasing the decode trace at every prefill was tested and rejected because it degraded the same benchmark to **41.56 t/s/u**, so the optimized path keeps persistent decode trace reuse.

Post-run cleanup audit found no leftover `vllm`, `EngineCore`, `api_server`, or `run_vllm_server` serving process; only the multigoal orchestrator itself remained.

## Profiler And Optimize Checklist

No Tracy, `tt-perf-report`, live-server device profiler, or `TT_METAL_DEVICE_PROFILER` collection was run for optimized vLLM serving. The selected vLLM/optimize skill contract intentionally disables live serving profiler collection on T3K; serving evidence is the same-harness benchmark plus trace/async/static contract checks. Existing device-op context comes from optimized full-model reduced artifacts under `doc/optimized_full_model/`, including LM-head/sampling evidence in `perf/reduced_1layer_lmhead8192_token_out_perf_report.csv`.

Relevant optimized-vLLM checklist items completed:

- Real vLLM TT plugin measured path through `tt/generator_vllm.py`.
- Full vLLM sampling tests, qualitative outputs, degenerate scan, and serving benchmark pass.
- Before/after same-harness TTFT, ITL, aggregate throughput, and per-user decode t/s/u recorded.
- Decode path uses nonblocking traced replay and async readback split.
- On-device sampling stays enabled with `sample_on_device_mode=all`; host sampling fallback is rejected.
- Full-model greedy split-sampling contract is reused through `SamplingGenerator` and `tt_out_tok` feedback; force-argmax and full-vocab host readback were rejected.
- Persistent scheduler inputs are reused, with token/current-position identity, unchanged-page-table no-copy behavior, and changed-page-table in-place refresh covered by contract checks.
- Runtime fallback and process cleanup audits are clean except for the documented request-boundary allocator warning.

## Limitations

- Serving is currently batch-1 / `max_num_seqs=1`.
- Prefix caching is not implemented.
- Async scheduler overlap is intentionally disabled.
- One conservative active-trace allocator warning remains at request-boundary prefill; removing it by releasing the decode trace per prefill costs too much decode throughput.
- Live vLLM profiler evidence is intentionally absent per the selected skill guidance; non-serving full-model profiler artifacts provide device-op context.
