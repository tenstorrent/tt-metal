# Optimized vLLM Work Log

## 2026-08-20

Scope: optimize completed `Qwen/Qwen3.6-35B-A3B` vLLM TT serving in place,
starting from the vLLM integration stage and the selected datatype-sweep
configuration. No context reduction was made; `max_model_len=262144` matches
`doc/context_contract.json`.

Implementation:

- Added async decode overlap capability in the Qwen vLLM adapter and kept
  `decode_forward(..., read_from_device=False)` on device. The plugin performs
  the required read at the async boundary.
- Reworked traced decode replay to use persistent token/current-position,
  RoPE, page-table, cache, and sampler tensors. Steady replay calls
  `ttnn.execute_trace(..., blocking=False)` and feeds the sampled device token
  back into the next token input without host readback.
- Reused the full-model split LM-head/sampling contract for decode. No host
  argmax, full-logits readback, or generic slow sampling path is part of the
  measured `sample_on_device_mode=all` path.
- Added adapter audit counters gated by `QWEN36_VLLM_AUDIT_PATH` to prove
  async decode, trace replay, nonblocking execute, changed-only host refresh,
  changed/unchanged page-table behavior, prefill sampling boundaries, and
  cleanup behavior.
- Added active-width decode cache support for idle single-user serving. The
  adapter slices linear-attention state and page tables to the active prefix
  width, commits or discards the active cache at prefill/trace release
  boundaries, and preserves the parent cache.
- Added a scheduler-owned `force_full_decode_width` path in the vLLM TT plugin.
  The scheduler sets a sticky full-width decode flag while there are waiting
  requests, multiple running requests, pending prefill, or multi-request
  scheduling. The flag resets once the scheduler is fully idle.
- Preserved non-aligned prompt support through padded internal execution and
  output slicing. Added explicit optimized serving evidence for prompt token
  length `26`.
- Extended static/unit coverage in `tests/test_generator_vllm.py` for async
  capability, no host fallback, stale token/current-position refresh,
  page-table unchanged hits, inactive live page-table row detection,
  active-cache lifecycle, and full-width burst guard behavior.

Primary benchmark command:

```bash
QWEN36_VLLM_AUDIT_PATH=models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/artifacts/after/vllm_trace_audit_benchmark_selective_cleanup.json \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --block-size 32 \
  --port 8024 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":384000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args=--async-scheduling
```

Result:

- Primary single-user `128/128/1`, concurrency `1`, temp `0.0`:
  TTFT `6274.1 ms`, TPOT mean `60.7 ms`, ITL P50/P99
  `57.3/58.1 ms`, aggregate output throughput `9.1520 tok/s`,
  TPOT-derived decode `16.4688 t/s/u`.
- Secondary CI burst `100/100/32`, unbounded concurrency, temp `0.0`:
  TTFT P50/P99 `155431.5/155432.3 ms`, TPOT mean `1000.1 ms`,
  ITL P50/P99 `915.1/2443.9 ms`, aggregate output throughput
  `12.8130 tok/s`, TPOT-derived decode `0.9999 t/s/u`.
- Raw logs:
  `artifacts/after/run_vllm_server_optimized_benchmark_selective_cleanup.log`,
  `artifacts/after/vllm_benchmark_optimized.log`, and
  `artifacts/after/vllm_ci_serving_benchmark_optimized.log`.

Before comparison, same runner shape and TT config:

- Primary single-user `128/128/1`, concurrency `1`, temp `0.0`:
  TTFT `7517.3 ms`, TPOT mean `945.3 ms`, ITL P50/P99
  `919.2/1232.0 ms`, aggregate output throughput `1.0034 tok/s`,
  TPOT-derived decode `1.0579 t/s/u`.
- Secondary CI burst `100/100/32`, unbounded concurrency, temp `0.0`:
  TTFT P50/P99 `156247.2/156248.2 ms`, TPOT mean `1003.5 ms`,
  ITL P50/P99 `919.2/3851.7 ms`, aggregate output throughput
  `12.7538 tok/s`, TPOT-derived decode `0.9965 t/s/u`.

Sampling and qualitative command:

```bash
QWEN36_VLLM_AUDIT_PATH=models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/artifacts/after/vllm_trace_audit_sampling_qualitative_fixed.json \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative \
  --sampling-profile full \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --block-size 32 \
  --port 8026 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":384000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args=--async-scheduling
```

Result: `72 passed, 1 skipped` in
`artifacts/after/sampling_tests_optimized.log`; default qualitative output was
written to `artifacts/after/vllm_qualitative_outputs_optimized.json`.

No-thinking and non-aligned checks:

```bash
QWEN36_VLLM_AUDIT_PATH=models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_vllm/artifacts/after/vllm_trace_audit_non_aligned_no_think.json \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --block-size 32 \
  --port 8027 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":384000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args=--async-scheduling
```

Against that server:

- `run_no_think_chat_qualitative.py` produced `12` records in
  `artifacts/after/vllm_chat_no_think_qualitative_outputs_optimized_vllm.json`.
- A direct chat request with prompt token length `26` returned HTTP `200` in
  `artifacts/after/non_aligned_prompt_check_optimized_vllm.json`.
- The no-thinking checker conversion used
  `build_no_think_controls.py --skip-tt` and wrote
  `artifacts/after/no_think_controls_optimized/vllm_chat_no_think_checker_outputs.json`.
- Degenerate-output checks passed for default and no-thinking outputs:
  `artifacts/after/degenerate_output_report_default_vllm.json` and
  `artifacts/after/degenerate_output_report_no_think_vllm.json`.

Local tests and static checks:

```bash
python_env/bin/python -m py_compile \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py
```

Result: passed.

```bash
python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_generator_vllm.py -q
```

Result: `9 passed`.

Trace/async audit summary:

- Benchmark: `decode_forward_read_from_device_false=226`,
  `async_decode_reads=226`, `decode_trace_replays=222`,
  `execute_trace_blocking_false=224`, `steady_device_feedback_replays=222`,
  `trace_input_host_refreshes=2`, `page_table_unchanged_hits=224`.
- Sampling/qualitative: `decode_forward_read_from_device_false=4074`,
  `async_decode_reads=2878`, `decode_trace_replays=2769`,
  `execute_trace_blocking_false=2821`, `steady_device_feedback_replays=2769`,
  `trace_input_host_refreshes=52`, `page_table_unchanged_hits=3917`.
- Non-aligned/no-thinking: `decode_forward_read_from_device_false=535`,
  `async_decode_reads=274`, `decode_trace_replays=260`,
  `execute_trace_blocking_false=267`, `steady_device_feedback_replays=260`,
  `trace_input_host_refreshes=7`, `page_table_unchanged_hits=528`.

Device and cleanup notes:

- No Tracy, tt-perf-report, live-server device profiler, adapter profiler, or
  ReadDeviceProfiler was collected for this stage.
- After the long sampling/qualitative run, one immediate serve-only startup
  failed with an active Ethernet core reset timeout. `tt-smi -r all` cleared
  the device state and the identical no-thinking/non-aligned server command
  then completed successfully.
- Final server shutdown was clean. The process cleanup check matched only the
  `pgrep` command itself for
  `run_vllm_server|vllm.entrypoints|api_server|VLLM::EngineCore|bench serve`.

Limitations:

- The CI burst profile is prefill/admission dominated and is capacity evidence,
  not headline decode t/s/u.
- The plugin intentionally keeps forced-full-width decode during bursty
  scheduler states to preserve cache/page-table correctness. The optimized
  narrow path is used for the idle single-user serving profile.
- The no-thinking qualitative script caps outputs at 64 tokens, so some
  accepted records are naturally truncated.

Stage review:

- Independent review returned `clean-pass`, stage-review subagent
  `01a02061-aee6-7102-84ef-3e11c325dc66`.
