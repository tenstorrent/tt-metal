# google/gemma-4-12B Optimized vLLM Serving

Batch-1 T3K serving result: prefill TTFT P50/P99 is 370.62 ms / 370.62 ms on the real vLLM TT plugin path; decode ITL P50/P99 is 41.92 ms / 42.27 ms, aggregate output throughput is 22.47 tok/s, and mean per-user decode throughput is 23.85 t/s/u. The batch-1 workload is `prompt_len=128`, `output_len=128`, `num_requests=1`, `concurrency=1`, `max_num_seqs=1`, `max_model_len=4096`, T3K, `sample_on_device_mode=all`, `trace_region_size=100000000`, and `fabric_config=FABRIC_1D_RING`.

This stage starts from the completed vLLM integration and optimizes the serving path in place. The measured path is `python -m models.common.readiness_check.run_vllm_server` launching vLLM with the TT plugin and the adapter at `models/autoports/google/gemma-4-12B/tt/generator_vllm.py`; it is not a standalone generator benchmark or eager fallback.

## Status

| Area | Result |
| --- | --- |
| vLLM sampling | Full profile passed: `71 passed, 1 skipped, 16 warnings in 318.13s` |
| Qualitative output | Serving completes but text quality remains poor and repetitive; this is tracked as model-quality/readiness limitation, not a crash/logprob/sampling failure |
| Async decode | Enabled and exercised by vLLM: server log reports `TTScheduler, async_scheduling=True` and `Asynchronous scheduling is enabled` |
| Trace replay | Persistent trace replay uses `ttnn.execute_trace(..., blocking=False)` for async decode reuse |
| On-device sampling | `sample_on_device_mode=all` and server log shows `models.common.sampling.tt_sampling:forward - Forcing argmax sampling`; no host greedy/top-1 argmax path is used by the measured serving run |
| Stale inputs | Focused trace-refresh test passed for changed token/current-position values and changed plus unchanged page tables |
| Cleanup | Server terminated cleanly after successful runs; no leftover `vllm.entrypoints`, `api_server`, or `EngineCore` process remained |

## Serving Metrics

Main before/after workload: `prompt_len=128`, `output_len=128`, `num_requests=32`, `concurrency=8`, `max_num_seqs=1`, `max_model_len=4096`, T3K, `sample_on_device_mode=all`, `trace_region_size=100000000`, `fabric_config=FABRIC_1D_RING`.

| Metric | Before vLLM integration | Optimized vLLM | Change |
| --- | ---: | ---: | ---: |
| Elapsed | 182.26 s | 174.77 s | 4.1% faster |
| TTFT P50 | 40003.67 ms | 38355.40 ms | 4.1% faster |
| TTFT P99 | 40203.03 ms | 38548.84 ms | 4.1% faster |
| ITL P50 | 43.95 ms | 41.92 ms | 4.6% faster |
| ITL P99 | 88.22 ms | 125.89 ms | 42.7% slower |
| Aggregate output throughput | 21.30 tok/s | 21.79 tok/s | 2.3% faster |
| Mean per-user decode | 21.76 t/s/u | 22.28 t/s/u | 2.4% faster |

Optimized full-model comparison: `doc/optimized_full_model/README.md` reports traced on-device full-wrapper decode at 23.08 t/s/u, 43.32 ms/token. The optimized vLLM serving path reaches 22.28 t/s/u on the 32-request workload and 23.85 t/s/u on batch 1. The comparable full-model gap is about 3.5% on the 32-request vLLM mean per-user metric, so no material scheduler/plugin gap remains for this `max_num_seqs=1` serving shape.

## Implementation

The vLLM adapter advertises async decode only after the path is exercised by vLLM:

- `generator_vllm.py` sets `model_capabilities["supports_async_decode"] = True`.
- `decode_forward(..., read_from_device=False)` returns TTNN device tensors and passes `async_decode=True` into the generator.
- `read_decode_output(..., async_read=True)` performs the deferred `.cpu(blocking=False)` transfer and returns the host object plus a TTNN event for the plugin boundary.
- `process_decode_output_host(...)` only formats host-side decode results into vLLM token/logprob structures.

The generator trace path now distinguishes capture from steady-state replay:

- First capture/compile still blocks and synchronizes.
- Trace reuse refreshes only the required token, current-position, cache-position, RoPE/page-table, cache, and sampler inputs for the request.
- Async serving replay calls `ttnn.execute_trace(self.mesh_device, state["id"], cq_id=0, blocking=False)`.
- Required synchronization/readback is deferred to the plugin async boundary instead of being forced inside decode replay.

## Validation

Sampling:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --sampling-profile full \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args '--chat-template models/autoports/google/gemma-4-12B/doc/vllm_integration/simple_chat_template.jinja --generation-config vllm'
```

Result: serving, full sampling, qualitative prompts, and benchmark completed. Sampling result was `71 passed, 1 skipped`; the skip is `test_chat_logprobs_all_vocab`. There were no reproducibility-only failures and no correctness, logprob, crash, or gibberish failures in the sampling suite.

Stale trace-input coverage:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py::test_decode_trace_refreshes_token_position_and_page_table_inputs \
  --tb=short --timeout=1200
```

Result: `1 passed, 3 warnings in 7.33s`. The first attempt hit an ARC startup timeout before model execution; after `tt-smi -r`, the focused test passed. The test captures a one-layer trace, replays with changed token/current-position and unchanged page table, then replays with changed token/current-position and swapped page-table entries.

Batch-1 serving:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args '--chat-template models/autoports/google/gemma-4-12B/doc/vllm_integration/simple_chat_template.jinja --generation-config vllm' \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 1 \
  --benchmark-concurrency 1
```

Result: server ready after about 50 s, `Requests: 1 completed in 5.7s`, TTFT P50/P99 370.6 ms / 370.6 ms, ITL P50/P99 41.9 ms / 42.3 ms, aggregate output 22.5 tok/s, mean per-user decode 23.9 t/s/u.

## Profiler

`tt-perf-report` is installed, but serving decode could not be profiled with the device profiler enabled. The batch-1 profiled server launch used the same workload and TT config with:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_vllm/profiler_batch1 \
python -m models.common.readiness_check.run_vllm_server --stages serve,benchmark ...
```

The server failed before readiness. Exact failure evidence:

- `artifacts/profiler_batch1_failed_readiness_vllm/server.log`: `EngineCore failed to start`.
- Same log: `RuntimeError: Timeout waiting for Ethernet core service remote IO request`.
- `profiler_batch1/generated/inspector/` contains only startup inspector YAMLs.
- No profiler ops CSV or log file was produced, so there was no valid `tt-perf-report` input.

Fallback timing evidence is the unprofiled real vLLM benchmark JSON in `artifacts/batch1_readiness_vllm/vllm_benchmark.json` and the full optimized serving benchmark in `artifacts/after_readiness_vllm/vllm_benchmark.json`.

## Runtime Audit

No unnecessary host fallback was found in the measured serving path:

- vLLM owns the serving cache; the adapter does not assume standalone generator cache ownership.
- The measured decode path returns TTNN tensors across `decode_forward(..., read_from_device=False)`.
- Sampling stays on device with `sample_on_device_mode=all`; the log shows TT sampler argmax activity rather than host greedy/top-1 fallback.
- Full logits are not read back for the measured decode benchmark.
- The async boundary performs the minimal deferred read required by the vLLM plugin contract.
- `pgrep -af 'vllm.entrypoints|VLLM::EngineCore|api_server|EngineCore'` after successful runs found no live server process except the `pgrep` command itself.

## Optimization Checklist

| Item | Evidence |
| --- | --- |
| Real vLLM TT plugin path | `run_vllm_server` launched `vllm.entrypoints.openai.api_server` with TT plugin config and `generator_vllm.py` adapter |
| Warm serving metrics before/after | Same 32-request workload preserved in `artifacts/before_readiness_vllm/` and `artifacts/after_readiness_vllm/` |
| Async decode | `supports_async_decode=True`, server log `async_scheduling=True`, deferred read in `read_decode_output(async_read=True)` |
| Nonblocking trace replay | Trace reuse calls `ttnn.execute_trace(..., blocking=False)` when async decode is requested |
| On-device sampling | `sample_on_device_mode=all`, TT sampler `Forcing argmax sampling` in server log |
| Stale-input coverage | Focused test covers changed token/current-position values and changed/unchanged page tables |
| Profiler effort | Device-profiler serving attempt preserved with exact startup failure and no ops CSV; fallback benchmark timings preserved |
| Cleanup | Successful server runs terminate cleanly and leave no vLLM/EngineCore process |

## Artifacts

- Before full serving run: `artifacts/before_readiness_vllm/`
- Optimized full serving run: `artifacts/after_readiness_vllm/`
- Batch-1 serving run: `artifacts/batch1_readiness_vllm/`
- Failed profiler serving run: `artifacts/profiler_batch1_failed_readiness_vllm/server.log`
- Profiler startup inspector files: `profiler_batch1/generated/inspector/`
- Current restored readiness directory: `../../readiness_vllm/`

## Limitations

- `max_num_seqs=1` only; larger serving batch sizes remain outside this adapter's validated path.
- Prefix caching is disabled.
- `test_chat_logprobs_all_vocab` is skipped by the plugin suite.
- Qualitative text remains poor, repetitive, and sometimes prompt-contaminated. That limitation predates this optimized-vLLM stage and is not fixed by async trace scheduling.
- The profiled serving launch failed during TT device initialization with the profiler enabled, before decode traffic. The exact UMD timeout is preserved, but serving decode lacks a usable `tt-perf-report` CSV for this stage.
