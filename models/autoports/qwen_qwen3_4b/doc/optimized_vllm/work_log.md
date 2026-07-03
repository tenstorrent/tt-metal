# Qwen3-4B Optimized vLLM Work Log

## 2026-07-03

- Started from completed vLLM integration in `doc/vllm_integration` and selected datatype-sweep config `baseline_bfp4_lofi_bf16kv_bf16ccl`.
- Confirmed context contract remains `40960` in `doc/context_contract.json`; no context, benchmark, or eval length reduction is introduced here.
- Confirmed serving artifacts use the TT vLLM plugin path through `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py` with `TT_QWEN3_TEXT_VER=autoport_qwen3_4b`.
- Confirmed the runner default merges `sample_on_device_mode=all` into `--additional-config` even when the command-line `--tt-config` only names trace region and fabric config.
- Confirmed final server log reports `sample_on_device_mode=all` and `async_scheduling=True`.
- Clarified that the command-line `--tt-config` omits `sample_on_device_mode` because `run_vllm_server.py` merges its default `{"sample_on_device_mode": "all"}` into the server `--additional-config`; final server log line 15 confirms the merged config.
- Classified `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json` as a stale pre-final debug artifact from the TP4 sampler investigation. It is excluded from final qualitative evidence because it contains repeated "Final Answer" text and "therm dynamics" spacing artifacts. The final qualitative verdict uses `readiness_vllm/vllm_qualitative_outputs.json`.
- Clarified cleanup evidence: the runner shut down the FastAPI server and UMD closed the device cluster; nanobind refcount leak diagnostics remain in the shutdown log and are not treated as proof of a live server or device hold.
- Added `doc/optimized_vllm/README.md` and `doc/optimized_vllm/perf_summary.json` to package the optimized-vLLM evidence separately from stage-08 vLLM integration.
- Did not collect Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or `ReadDeviceProfiler` evidence for this vLLM serving stage.
- Preserved the completed vLLM integration benchmark artifacts under `doc/optimized_vllm/before/`.
- First after rerun failed before model code while opening the mesh: `Device 0: Timed out while waiting for active ethernet core 31-25 to become active again`. Preserved the log under `doc/optimized_vllm/after_failed/server_active_eth_startup_failure.log`.
- Ran bounded recovery: `timeout 60 tt-smi -ls --local && timeout 180 tt-smi -r && timeout 60 tt-smi -ls --local`; all four P150 chips were visible after reset.
- Ran mesh smoke with `ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)`, which printed `MESH_SMOKE_OK`.
- Retried the same `run_vllm_server` command. The retry passed full sampling, qualitative, primary single-user benchmark, CI serving-burst benchmark, and clean server termination.
- Copied the after artifacts under `doc/optimized_vllm/after/`.
- Committed server-log artifacts are bounded extracts of the final raw server log because the repository hook rejects files over 500 KB. The extracts retain startup/config, `max_model_len=40960`, `sample_on_device_mode=all`, async scheduling, and shutdown/UMD cleanup evidence. Benchmark metrics remain sourced from the JSON benchmark artifacts.

## Commands

Final inherited run from `doc/vllm_integration`:

```bash
TT_QWEN3_TEXT_VER=autoport_qwen3_4b \
VLLM_PLUGINS=tt,tt_model_registry \
QWEN3_4B_AUTOPORT_DIR=/home/ubuntu/tt-metal/models/autoports/qwen_qwen3_4b \
PYTHONPATH=/home/ubuntu/vllm:$PYTHONPATH \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/qwen_qwen3_4b \
  --hf-model Qwen/Qwen3-4B \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 40960 \
  --block-size 32 \
  --sampling-profile full \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 134217728, "fabric_config": "FABRIC_1D_RING"}'
```

Non-aligned prompt-length check:

```bash
python_env/bin/vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-4B \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 37 \
  --random-output-len 8 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-filename models/autoports/qwen_qwen3_4b/readiness_vllm/non_aligned_prompt_len37_result.json \
  --temperature 0.0
```

Source/contract test:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short
```

## Metrics

Before primary single-user benchmark, 128 prompt tokens, 128 output tokens, 1 request, concurrency 1, temperature 0.0, ignore EOS:

- Artifact: `doc/optimized_vllm/before/vllm_benchmark.json`
- TTFT P50/P99/mean: `225.6/225.6/225.6 ms`
- TPOT mean/P99: `13.3/13.3 ms`
- ITL P50/P99/mean: `10.3/10.7/13.3 ms`
- Output throughput: `67.0 tok/s`
- TPOT-derived decode: `75.4 t/s/u`

After primary single-user benchmark, same workload:

- Artifact: `doc/optimized_vllm/after/vllm_benchmark.json`
- TTFT P50/P99/mean: `218.9/218.9/218.9 ms`
- TPOT mean/P99: `13.2/13.2 ms`
- ITL P50/P99/mean: `10.3/10.6/13.2 ms`
- Output throughput: `67.3 tok/s`
- TPOT-derived decode: `75.5 t/s/u`

Before CI serving-burst benchmark, 100 prompt tokens, 100 output tokens, 32 requests, burst admission, temperature 0.0, ignore EOS:

- Artifact: `doc/optimized_vllm/before/vllm_ci_serving_benchmark.json`
- TTFT P50/P99/mean: `1893.0/2869.6/2172.4 ms`
- TPOT mean/P99: `33.2/48.4 ms`
- ITL P50/P99/mean: `17.7/295.8/33.2 ms`
- Output throughput: `586.3 tok/s`
- TPOT-derived decode: `30.2 t/s/u`, secondary only

After CI serving-burst benchmark, same workload:

- Artifact: `doc/optimized_vllm/after/vllm_ci_serving_benchmark.json`
- TTFT P50/P99/mean: `1868.8/2837.4/2145.8 ms`
- TPOT mean/P99: `28.9/44.0 ms`
- ITL P50/P99/mean: `17.7/159.6/28.9 ms`
- Output throughput: `638.4 tok/s`
- TPOT-derived decode: `34.6 t/s/u`, secondary only

## Trace And Async Decisions

- `Qwen3ForCausalLM.model_capabilities["supports_async_decode"]` is `True`.
- `decode_forward(..., read_from_device=False, perform_device_sampling=True)` returns the generator's device token tensor from `prepare_token_out_decode` or `decode_next_token_traced`.
- The TT plugin calls `read_decode_output(..., async_read=True)` and later `process_decode_output_host(...)` through `TTAsyncDecodeController.finalize_decode`.
- `Qwen3Generator.decode_next_token_on_device` calls `self.model.execute_decode_trace()` and then sampled-token trace replay without host token readback.
- `Qwen3FullModel.execute_decode_trace` is the model-side nonblocking trace replay point.
- Page-table refresh is generation-tracked; unchanged page tables return the existing persistent device tensor.
- The final active-trace warning is classified as shutdown-adjacent in `readiness_vllm/trace_warning_audit.log`.

## Optimize Checklist Mapping

- Real measured path is vLLM TT plugin serving through `tt/generator_vllm.py`: covered by server log and plugin registration.
- Sampling tests, qualitative outputs, primary benchmark, and CI benchmark: artifacts exist under `readiness_vllm`.
- Context contract preserved: `max_model_len=40960` matches `doc/context_contract.json`.
- Non-aligned prompt support preserved: prompt length 37 check completed `1/1`.
- Async decode split implemented: adapter and plugin code paths inspected; source tests cover capability and generation forwarding.
- On-device sampling preserved for measured greedy path: `sample_on_device_mode=all`, TP4 greedy sampler, no adapter host argmax.
- Profiler collection skipped for vLLM stage: no vLLM-stage profiler artifacts added.
- Runtime cleanup: `doc/vllm_integration/work_log.md` records no live vLLM server or EngineCore worker after final run.
- Batch/concurrency coverage: CI serving-burst uses 32 requests with `max_num_seqs=32`.

## Limitations

- Prefix caching is disabled.
- On-device serving sampling is greedy-only for the performance path.
- Stochastic or host-only sampling-parameter compatibility is not used as the optimized serving performance path.
- `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json` is excluded as stale debug evidence from before the final sampler fix. It is intentionally not used for final qualitative status.
- Server shutdown logs include nanobind refcount leak diagnostics after device close. Cleanup status is limited to no live vLLM server or EngineCore worker holding devices.

## Commit Log

- Starting tt-metal commit: `9019b51e2bd Add Qwen3 4B vLLM integration`
- Starting vLLM commit: `de6c44fd89154bd800c8c947e7205876b93013e3 Add Qwen3 autoport vLLM TT support`
- Final local commits will be recorded after stage review and checkpoint creation.
