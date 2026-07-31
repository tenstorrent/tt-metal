# Optimized vLLM Work Log

## Scope

Goal: create the optimized-vLLM state for `google/gemma-4-12B` in the current `tt-metal` checkout. This starts from completed vLLM integration and optimizes the real vLLM TT plugin serving path through `models/autoports/google/gemma-4-12B/tt/generator_vllm.py`.

Applied skills: `vllm-integration`, `optimize`, and `tt-enable-tracing`. `autofix` was not invoked in this stage because no new unresolved subtle trace/cache serving correctness bug remained after the focused stale-input test passed. The earlier vLLM integration stage already used AutoFix for stale traced decode behavior; this stage added direct regression coverage instead of reopening decoder or full-model bringup.

Provenance:

- HF model: `google/gemma-4-12B`
- Hardware: Wormhole T3K, 8 devices
- Mesh/fabric: T3K, `FABRIC_1D_RING`
- vLLM adapter: `../../tt/generator_vllm.py`
- Generator trace path: `../../tt/generator.py`
- Starting vLLM integration report: `../vllm_integration/README.md`
- Optimized full-model comparison: `../optimized_full_model/README.md`

## Implementation Changes

Updated `tt/generator.py`:

- Added an `async_decode` flag from `decode_forward(...)` into `decode_forward_traced(...)`.
- Kept initial trace capture blocking and synchronized.
- Changed steady-state trace reuse to `ttnn.execute_trace(self.mesh_device, state["id"], cq_id=0, blocking=not async_decode)`.
- Skipped `ttnn.synchronize_device(...)` inside replay when async decode is requested, leaving synchronization to the vLLM plugin async boundary.

Updated `tt/generator_vllm.py`:

- Advertised `model_capabilities["supports_async_decode"] = True`.
- Routed `decode_forward(..., read_from_device=False)` to return device tensors and request async traced replay.
- Added safe TTNN-to-torch conversion for both device and host TTNN tensors.
- Added nonblocking deferred reads in `read_decode_output(..., async_read=True)` with `.cpu(blocking=False)`.
- Returned a TTNN event to match the vLLM TT plugin async contract.
- Kept `process_decode_output_host(...)` as host-result formatting only.

Updated `tests/test_optimized_full_model.py`:

- Added `test_decode_trace_refreshes_token_position_and_page_table_inputs`.
- The test uses a one-layer T3K generator to capture and replay the trace.
- It verifies changed token/current-position values, unchanged page table reuse, and changed page-table refresh.

## Before Baseline

Baseline artifact copied from the completed vLLM integration state to `artifacts/before_readiness_vllm/`.

Benchmark workload:

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

Result from `artifacts/before_readiness_vllm/vllm_benchmark.json`:

```json
{
  "elapsed_s": 182.25909815402701,
  "completed_requests": 32,
  "total_output_tokens": 3883,
  "ttft_p50_ms": 40003.67089919746,
  "ttft_p99_ms": 40203.02640926093,
  "itl_p50_ms": 43.950119987130165,
  "itl_p99_ms": 88.21595925837755,
  "output_throughput_tok_per_s": 21.304834926366638,
  "mean_per_request_decode_tps": 21.755547966443146
}
```

## Validation Commands

Syntax checks:

```bash
python -m py_compile \
  models/autoports/google/gemma-4-12B/tt/generator.py \
  models/autoports/google/gemma-4-12B/tt/generator_vllm.py

python -m py_compile \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py
```

Result: both commands passed.

Focused stale-input test:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_full_model.py::test_decode_trace_refreshes_token_position_and_page_table_inputs \
  --tb=short --timeout=1200
```

First attempt failed before model execution with an ARC startup timeout:

```text
ARC startup error at core 0-10 over NOC0 ... Timed out after 300000 ms
```

After board reset:

```bash
tt-smi -r
```

Retry result:

```text
1 passed, 3 warnings in 7.33s
```

Full optimized vLLM run:

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

Result:

```text
Server ready after ~50s
71 passed, 1 skipped, 16 warnings in 318.13s
Requests : 32 completed in 174.8s
TTFT     : P50=38355.4ms  P99=38548.8ms
ITL      : P50=41.9ms  P99=125.9ms
Output   : 21.8 tok/s aggregate
Per-user : 22.3 t/s/u
```

The final full run was copied to `artifacts/after_readiness_vllm/` and restored to `../../readiness_vllm/` at the end.

Batch-1 serving run:

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

Result:

```json
{
  "elapsed_s": 5.696068240329623,
  "completed_requests": 1,
  "total_output_tokens": 128,
  "ttft_ms": {
    "p50": 370.62496691942215,
    "p99": 370.62496691942215,
    "mean": 370.62496691942215
  },
  "itl_ms": {
    "p50": 41.916311252862215,
    "p99": 42.268207762390375,
    "mean": 41.923121527189345
  },
  "output_throughput_tok_per_s": 22.47164089322652,
  "mean_per_request_decode_tps": 23.852315247260385
}
```

## Async and Trace Evidence

Server log evidence from `artifacts/after_readiness_vllm/server.log`:

```text
plugin_config={'tt': {'sample_on_device_mode': 'all', 'trace_region_size': 100000000, 'fabric_config': 'FABRIC_1D_RING'}}
Scheduler class: TTScheduler, async_scheduling=True
Asynchronous scheduling is enabled.
models.common.sampling.tt_sampling:forward:428 - Forcing argmax sampling
```

Code evidence:

- `generator_vllm.py`: `supports_async_decode=True`.
- `generator_vllm.py`: `decode_forward(... read_from_device=False)` returns TTNN device tensors.
- `generator_vllm.py`: `read_decode_output(... async_read=True)` performs `.cpu(blocking=False)` and returns a TTNN event.
- `generator.py`: trace reuse calls `ttnn.execute_trace(..., blocking=not async_decode)`.

The plugin owns the event synchronization after the async boundary. Host conversion is limited to the deferred output read and final vLLM result formatting.

## Profiler Attempt

Command:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_vllm/profiler_batch1 \
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

Result: failed before server readiness. The wrapper detected `EngineCore failed to start` and terminated the server. Exact log lines are preserved in `artifacts/profiler_batch1_failed_readiness_vllm/server.log`:

```text
RuntimeError: Timeout waiting for Ethernet core service remote IO request.
```

`tt-perf-report` exists at `/localdev/moconnor/tt-metal/python_env/bin/tt-perf-report`, but there was no ops CSV or profiler log to process. The profiler output directory contains only:

```text
profiler_batch1/generated/inspector/kernels.yaml
profiler_batch1/generated/inspector/mesh_devices_log.yaml
profiler_batch1/generated/inspector/mesh_workloads_log.yaml
profiler_batch1/generated/inspector/programs_log.yaml
profiler_batch1/generated/inspector/startup.yaml
```

Fallback timing evidence is the successful unprofiled batch-1 and full serving benchmark JSON.

## Rejected Options

- Restarting decoder or full-model bringup: rejected because serving evidence did not show a concrete decoder/full-model correctness bug. The remaining text-quality problem already existed in vLLM integration.
- Host greedy/top-1 fallback: rejected because `sample_on_device_mode=all` keeps sampling on the TT sampler path and the server log confirms TT sampler argmax activity.
- Full-logits readback for serving decode: rejected for the measured benchmark path; async decode returns device tensors and only reads the minimal output required by the plugin contract.
- Treating profiler startup failure as decode evidence: rejected. The failure happened during mesh-device initialization before decode traffic, so it is documented as a profiler limitation and not as a serving decode profile.
- Marking the run blocked after profiler failure: rejected because unprofiled real vLLM timing and async/trace validation were still meaningful and completed.

## Cleanup

After successful vLLM runs:

```bash
pgrep -af 'vllm.entrypoints|VLLM::EngineCore|api_server|EngineCore' || true
```

Result: no live server process remained except the `pgrep` command itself. The failed profiler launch required `tt-smi -r`; the following unprofiled batch-1 serving run completed cleanly.

## Artifacts

- `README.md`: optimized-vLLM summary.
- `artifacts/before_readiness_vllm/`: baseline vLLM integration readiness artifacts.
- `artifacts/after_readiness_vllm/`: optimized full serving, sampling, qualitative, and benchmark artifacts.
- `artifacts/batch1_readiness_vllm/`: batch-1 serving benchmark artifacts.
- `artifacts/profiler_batch1_failed_readiness_vllm/server.log`: profiler startup failure evidence.
- `profiler_batch1/generated/inspector/`: profiler startup inspector YAMLs.
- `../../readiness_vllm/`: restored optimized full serving artifacts.
