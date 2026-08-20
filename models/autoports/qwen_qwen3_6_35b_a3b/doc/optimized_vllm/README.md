# Qwen3.6-35B-A3B Optimized vLLM

Status: `Qwen/Qwen3.6-35B-A3B` serves through the real TT vLLM plugin path on
P300C using
`models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py`. The optimized
headline path is the single-user vLLM serving workload
`128 input / 128 output / 1 request / max concurrency 1 / temperature 0.0 /
max_num_seqs 32 / max_model_len 262144 / block_size 32 / P300C /
sample_on_device_mode=all / trace_region_size=384000000 /
FABRIC_1D_RING / --async-scheduling`.

| Primary single-user vLLM serving path | Workload | TTFT P50/P99 ms | TPOT mean ms | ITL P50/P99 ms | Aggregate output tok/s | TPOT-derived decode t/s/u | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before optimization | `128/128/1`, concurrency `1`, temp `0.0` | `7517.3 / 7517.3` | `945.3` | `919.2 / 1232.0` | `1.0034` | `1.0579` | `artifacts/before/vllm_benchmark.json` |
| After optimization | `128/128/1`, concurrency `1`, temp `0.0` | `6274.1 / 6274.1` | `60.7` | `57.3 / 58.1` | `9.1520` | `16.4688` | `artifacts/after/vllm_benchmark_optimized.json` |

CI serving-burst is retained as secondary capacity/nightly-parity evidence, not
as the headline decode t/s/u:

| Secondary CI burst path | Workload | TTFT P50/P99 ms | TPOT mean ms | ITL P50/P99 ms | Aggregate output tok/s | TPOT-derived decode t/s/u | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before optimization | `100/100/32`, unbounded concurrency, temp `0.0` | `156247.2 / 156248.2` | `1003.5` | `919.2 / 3851.7` | `12.7538` | `0.9965` | `artifacts/before/vllm_ci_serving_benchmark.json` |
| After optimization | `100/100/32`, unbounded concurrency, temp `0.0` | `155431.5 / 155432.3` | `1000.1` | `915.1 / 2443.9` | `12.8130` | `0.9999` | `artifacts/after/vllm_ci_serving_benchmark_optimized.json` |

Full-model comparison for comparable `128/128` traced decode:
`doc/optimized_full_model/artifacts/perf_summary.json` reports optimized
full-model traced decode `17.4339 t/s/u` and TTFT `5895.7 ms`. Optimized vLLM
serving decode is `16.4688 t/s/u`, or `94.5%` of that full-model traced decode
rate, on the same mesh and precision family.

Summary/provenance: `artifacts/perf_summary.json`.

## Serving Path

| Contract | Final setting |
| --- | --- |
| Model | `Qwen/Qwen3.6-35B-A3B` |
| Adapter | `tt/generator_vllm.py::Qwen3_5MoeForConditionalGeneration` |
| Plugin path | `/localdev/vkovacevic/vllm/plugins/vllm-tt-plugin` |
| Hardware | local `2x2` Blackhole p300c mesh, `FABRIC_1D_RING` |
| Context | `max_model_len=262144`, matching `doc/context_contract.json` |
| vLLM scheduler | `--async-scheduling`, `max_num_seqs=32`, block size `32` |
| Sampling | `sample_on_device_mode=all`, full-model split sampling reused |
| Decode trace | persistent inputs, `ttnn.execute_trace(..., blocking=False)` |
| Async decode | `decode_forward(..., read_from_device=False)` returns device tensors; plugin reads at the async boundary |

The measured path has no adapter host greedy/top-1 argmax, full-logits
readback, generic slow sampling replacement, Python token readback/writeback
feedback loop, or per-token unchanged page-table rebuild. Prefill and decode
reuse the full-model generator LM-head/sampler contract.

## Trace And Async Evidence

Benchmark audit artifact:
`artifacts/after/vllm_trace_audit_benchmark_selective_cleanup.json`.

Key benchmark counters:

- `decode_forward_calls=226`; `decode_forward_read_from_device_false=226`.
- `read_decode_output_calls=226`; `async_decode_reads=226`;
  `process_decode_output_host_calls=226`.
- `decode_trace_captures=2`; `decode_trace_replays=222`;
  `execute_trace_blocking_false=224`.
- `steady_device_feedback_replays=222`; `trace_input_host_refreshes=2`.
- `page_table_device_refreshes=5`; `page_table_unchanged_hits=224`.
- `decode_trace_width_1=127`; `decode_trace_width_32=99`;
  `force_full_decode_width_true=99`.

Sampling and qualitative audit artifact:
`artifacts/after/vllm_trace_audit_sampling_qualitative_fixed.json`. It records
`decode_forward_calls=4074`, `decode_forward_read_from_device_false=4074`,
`async_decode_reads=2878`, `execute_trace_blocking_false=2821`,
`steady_device_feedback_replays=2769`, and `page_table_unchanged_hits=3917`.

The adapter uses active-width decode for steady single-user idle serving and a
scheduler-propagated forced-full-width flag while another request can interact
with live scheduler state. That keeps burst/page-table behavior correct without
trading away the optimized single-user decode path.

## Correctness Evidence

- Adapter tests:
  `python_env/bin/python -m pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_generator_vllm.py -q`
  passed (`9 passed`). The tests cover async capability flags, no host
  sampling fallback, changed token/current-position refresh, changed/unchanged
  page tables, inactive page-table row detection, active-cache commit/discard,
  and the scheduler full-width guard.
- vLLM sampling suite:
  `artifacts/after/sampling_tests_optimized.log`, `72 passed, 1 skipped`.
- Default qualitative output:
  `artifacts/after/vllm_qualitative_outputs_optimized.json`; degeneracy report
  `artifacts/after/degenerate_output_report_default_vllm.json`; result:
  no degenerate output detected.
- No-thinking qualitative output:
  `artifacts/after/vllm_chat_no_think_qualitative_outputs_optimized_vllm.json`
  with `12` records; checker artifact
  `artifacts/after/no_think_controls_optimized/vllm_chat_no_think_checker_outputs.json`;
  degeneracy report
  `artifacts/after/degenerate_output_report_no_think_vllm.json`; result:
  no degenerate output detected.
- Non-aligned serving request:
  `artifacts/after/non_aligned_prompt_check_optimized_vllm.json` records a
  direct chat request with prompt token length `26`, non-divisible by
  `32`, `64`, and `128`, and HTTP `200`.

## Optimize Checklist Evidence

- Measured before and after with the same `run_vllm_server` workload,
  generation mode, sampling mode, mesh, `max_num_seqs`, `max_model_len`,
  block size, and TT config.
- Preserved `doc/context_contract.json` `supported_context=262144`; no context
  or eval length was reduced.
- Preserved valid non-aligned prompt-length serving support.
- Kept sample-on-device serving with the full-model split sampler; the
  optimized path does not use force-argmax or full-logits host readback.
- Reused persistent token, position, RoPE, page-table, cache, and sampler trace
  inputs; refreshed host inputs only when reset/scheduler state required it.
- Used benchmark JSON, sampling tests, stale-input unit tests, async audit
  counters, qualitative outputs, and process cleanup evidence. No Tracy,
  tt-perf-report, live-server device profiler, adapter profiler, or
  ReadDeviceProfiler data was collected in this stage.

## Rejected Options

- Aligned-only decode specialization was rejected; non-aligned prompt serving
  remains part of the accepted path.
- Host greedy/top-1 argmax, force-argmax, full-logits readback, and generic
  slow sampling were rejected because they violate the serving sampling
  contract and would hide sampler/LM-head performance.
- Always-narrow decode during scheduler bursts was rejected after burst
  correctness investigation. The final plugin forces full-width decode while
  there are waiting/multiple/pending-prefill requests, then returns to narrow
  active-width decode when the scheduler is idle.
- Profiling tools were rejected for this stage per the goal contract; evidence
  comes from runner JSON, tests, and explicit adapter audit counters.

## Cleanup And Limitations

Final serve-only cleanup was clean: the server terminated normally and
`pgrep -af 'run_vllm_server|vllm.entrypoints|api_server|VLLM::EngineCore|bench serve'`
matched only the `pgrep` command itself.

One immediate restart after the long sampling/qualitative run failed to open
the mesh because an active Ethernet core did not reset. `tt-smi -r all` cleared
the device state and the same no-thinking/non-aligned server command completed
successfully afterward. The final optimized server shutdown left no vLLM or
EngineCore process behind.

The CI burst profile remains prefill/admission dominated and is intentionally
reported only as secondary capacity evidence.
