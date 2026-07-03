# Qwen3-4B Optimized vLLM Serving

## Headline

Primary single-user vLLM TT serving through `tt/generator_vllm.py`:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | Decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 prompt tokens, 128 output tokens, 1 request, concurrency 1, temperature 0.0, ignore EOS, `max_model_len=40960`, `max_num_seqs=32`, `sample_on_device_mode=all` | 218.9 / 218.9 ms | 13.2 / 13.2 ms | 10.3 / 10.6 ms | 67.3 output tok/s | 75.5 |

CI serving-burst evidence is secondary:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | TPOT-derived t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 prompt tokens, 100 output tokens, 32 requests, burst admission, temperature 0.0, ignore EOS, `max_model_len=40960`, `max_num_seqs=32`, `sample_on_device_mode=all` | 1868.8 / 2837.4 ms | 28.9 / 44.0 ms | 17.7 / 159.6 ms | 638.4 output tok/s | 34.6 |

The headline decode value is `1000 / mean_tpot_ms` from the primary single-user profile. The CI burst profile is only capacity and nightly-parity evidence.

## Serving Contract

- Model: `Qwen/Qwen3-4B`
- Autoport: `models/autoports/qwen_qwen3_4b`
- Adapter: `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py`
- vLLM plugin registration: `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, selected with `TT_QWEN3_TEXT_VER=autoport_qwen3_4b`
- Mesh: `P150x4`, tensor parallel degree 4
- Context: `40960`, matching `doc/context_contract.json`
- Selected datatype config: `doc/datatype_sweep/selected_precision_config.json`, `baseline_bfp4_lofi_bf16kv_bf16ccl`
- Sampling mode: `sample_on_device_mode=all`
- Prefix caching: disabled

The measured path is the real TT vLLM plugin path. vLLM calls `Qwen3ForCausalLM.decode_forward(..., read_from_device=False, perform_device_sampling=True)`, the adapter returns device tensors, the plugin defers the read through `read_decode_output(..., async_read=True)`, and host formatting is limited to `process_decode_output_host(...)` after the async boundary.

## Trace And Sampling

Serving reuses the full-model traced token-out path and TP4 greedy sampler. The generator submits decode replay with `ttnn.execute_trace(..., blocking=False)` and sampling replay also uses nonblocking trace execution. The sampled token is written into the persistent decode token input through `tt_out_tok`; the serving path does not use host greedy argmax, full-logits readback, or eager adapter-side sampling for the measured greedy benchmark.

Persistent trace inputs cover token, current position, RoPE position, page table, KV cache, and sampler output token state. The adapter tracks page-table generations and only refreshes the device page table when the host table changes. Contract tests cover changed token/current-position and changed/unchanged page-table generation forwarding.

## Evidence

- Full sampling suite: `readiness_vllm/sampling_tests.log`, `72 passed`, `1 skipped`
- Qualitative outputs: `readiness_vllm/vllm_qualitative_outputs.json`
- Degenerate-output check: recorded in `doc/vllm_integration/work_log.md`, no findings
- Primary benchmark: `readiness_vllm/vllm_benchmark.json`
- CI serving-burst benchmark: `readiness_vllm/vllm_ci_serving_benchmark.json`
- Non-aligned prompt support: `readiness_vllm/non_aligned_prompt_len37_result.json`, prompt length 37, output length 8, completed `1/1`
- Adapter/stale-input tests: `readiness_vllm/adapter_runtime_tests.log`, mirrored by `tests/test_full_model_contract.py`
- Trace lifecycle audit: `readiness_vllm/trace_warning_audit.log`
- vLLM server evidence log: `readiness_vllm/server.log`; this is a bounded extract of the final raw server log containing startup/config, async/context/on-device sampling signposts, and shutdown/cleanup lines.
- vLLM-stage perf summary: `doc/optimized_vllm/perf_summary.json`
- Preserved before artifacts: `doc/optimized_vllm/before/`
- Optimized after artifacts: `doc/optimized_vllm/after/`
- Failed first after attempt and recovery evidence: `doc/optimized_vllm/after_failed/server_active_eth_startup_failure.log`

Excluded stale/debug artifact:

- `readiness_vllm/vllm_qualitative_outputs_on_device_256_debug.json` is a pre-final debug capture from the TP4 sampler investigation. It contains visible degraded output, including repeated "Final Answer" text on the French prompt and "therm dynamics" spacing artifacts. It is not part of the final optimized-vLLM qualitative verdict. The final verdict uses `readiness_vllm/vllm_qualitative_outputs.json`, which passed the degenerate-output check.

No Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or `ReadDeviceProfiler` evidence was collected for this vLLM stage. Device-time and roofline fields in `perf_summary.json` are intentionally `null` for `vllm_serving_profiler_disabled_to_protect_hardware`.

## Before And After

The before artifacts are the completed vLLM integration measurements preserved before the optimized-vLLM rerun. The after artifacts are from the same `run_vllm_server` workload, generation mode, `max_num_seqs`, `max_model_len`, mesh, sampling mode, and TT config after the optimized-vLLM packaging and vLLM comment cleanup.

| Profile | Baseline artifact | Optimized artifact | Baseline decode | Optimized decode |
| --- | --- | --- | ---: | ---: |
| Primary 128/128/1 | `doc/optimized_vllm/before/vllm_benchmark.json` | `doc/optimized_vllm/after/vllm_benchmark.json` | 75.4 t/s/u | 75.5 t/s/u |
| CI 100/100/32 | `doc/optimized_vllm/before/vllm_ci_serving_benchmark.json` | `doc/optimized_vllm/after/vllm_ci_serving_benchmark.json` | 30.2 t/s/u | 34.6 t/s/u |

The first after attempt failed before model code with an active Ethernet core heartbeat timeout while opening the mesh. The failure log is preserved in `doc/optimized_vllm/after_failed/`; bounded `tt-smi` reset/list recovery and a 1x4 mesh smoke succeeded, then the same after command passed.

## Full-Model Comparison

`doc/optimized_full_model/perf_summary.json` reports steady traced token-out decode at `96.91 t/s/u` (`10.32 ms/token`) and traced teacher-forcing decode at `63.95 t/s/u`. The primary vLLM serving path reports `75.48 t/s/u` from mean TPOT, including serving orchestration, async output handling, on-device sampling, token feedback, and API request handling.

## Rejected Options And Limits

- Host greedy/top-1 argmax and full-logits readback are rejected for the measured path; they remain compatibility/debug behavior only outside the on-device greedy serving benchmark.
- Force-argmax sampling is rejected for the measured path because `Sampling1D` force-argmax uses a trace-unsafe output-tensor argmax path. The TP4 greedy sampler is used instead.
- Prefix caching remains disabled.
- Active-trace allocation warning after the final non-aligned rerun is classified in `readiness_vllm/trace_warning_audit.log`: no decode signposts or trace executions follow the warning before server idle/shutdown.
- Server cleanup means no live vLLM server or EngineCore worker remained holding devices after the runner shutdown. The server log still includes nanobind refcount leak diagnostics during interpreter teardown; those are recorded as shutdown diagnostics, not evidence of a live serving process.
