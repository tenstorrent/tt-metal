# Llama 3.1 8B Instruct Optimized vLLM

Status: optimized-vLLM evidence collected on 2026-06-15.

| Batch-1 serving metric | Before vLLM integration baseline | Optimized vLLM |
| --- | ---: | ---: |
| Sampling status | 41 passed, 22 skipped, 9 xfailed | 41 passed, 22 skipped, 9 xfailed |
| Qualitative verdict | coherent/on-topic, no degeneracy | coherent/on-topic, no degeneracy |
| Prefill TTFT P50 | 81.10 ms | 80.98 ms |
| Prefill TTFT P99 | 1047.01 ms | 990.80 ms |
| ITL P50 | 15.77 ms | 15.75 ms |
| ITL P99 | 25.09 ms | 22.71 ms |
| Aggregate output throughput | 51.84 tok/s | 52.07 tok/s |
| Mean per-user decode | 53.90 t/s/u | 54.10 t/s/u |

Batch-1 benchmark workload: prompt length 128, output length 128, 32
requests, concurrency 1, `max_num_seqs=1`, `max_model_len=1024`, T3K 1x8,
`FABRIC_1D_RING`, `trace_mode=decode_only`, `sample_on_device_mode=all`.

## What Changed

- `tt/generator.py` now always submits serving decode trace replay with
  `ttnn.execute_trace(..., blocking=False)`. The only synchronize left in that
  block is before first sampling-trace capture when a sampling trace is not yet
  ready.
- `tt/generator_vllm.py` now records the sampled token returned by on-device
  prefill/decode and refreshes the persistent decode token input only when the
  next host scheduler token differs. The unchanged-token steady case keeps
  `tt_out_tok` device feedback and performs no token copy.
- Added `vllm_async_contract_probe.py`, which drives the actual vLLM adapter
  API with a vLLM-owned KV cache and verifies async submit/read/host formatting,
  nonblocking model trace replay, changed-token refresh, changed-position
  refresh, unchanged page-table no-copy, and changed page-table copy-once.

## Serving Commands

Before artifact, archived from the completed vLLM integration run:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --port 8001
```

After benchmark command:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --port 8001
```

Final sampling and qualitative command:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --sampling-profile full \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --port 8001
```

Degeneracy check:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --missing-artifacts critical \
  --scope vllm
```

## Async And Trace Contract

Adapter capabilities:

- `supports_prefix_caching=False`
- `supports_async_decode=True`
- `supports_sample_on_device=True`
- `tt_async_decode_allows_overlap=False`

The measured server used the TT plugin path through
`models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator_vllm.py`.
The plugin server logs show `sample_on_device_mode=all`,
`trace_mode=decode_only`, and `async_scheduling=True`.

`vllm_async_contract_probe.json` status is `pass`. It proves:

- `decode_forward(..., read_from_device=False)` returns device tensors.
- `read_decode_output(..., async_read=True)` records one deferred event.
- `process_decode_output_host(...)` only formats host token results.
- Every model trace replay in the probe used the nonblocking replay counter.
- Unchanged page tables do not copy per token.
- Changed page tables copy once and are marked changed-only.
- Changed host token values refresh the persistent token tensor once.
- Changed current positions refresh token, position, RoPE, and page table.
- Greedy serving uses split sampling with `force_argmax=False`.

## Sampling And Quality

Final full sampling profile:

```text
41 passed, 22 skipped, 9 xfailed, 16 warnings in 120.71s
```

The skipped cases are host-compatibility or full/top-k logprob features that
are intentionally unavailable under strict `sample_on_device_mode=all`.
The xfailed cases are the same no-seed variety/top-k variety checks recorded in
the vLLM integration stage.

Qualitative outputs are coherent and on topic for haiku, learning explanation,
story continuation, thermodynamics, translation, and Fibonacci prompts. The
known minor leading-token duplication (`a a`, `The The`) remains, but the
degeneracy check reported:

```text
No degenerate output detected.
```

## Full-Model Comparison

Optimized full-model no-readback token-out evidence:

- TTFT: `629.73 ms`
- Steady traced replay: `70.58 t/s/u`
- Steady traced replay latency: `14.17 ms/token`
- Sampled-token readbacks: `0`

Optimized vLLM after evidence:

- Mean per-user decode: `54.10 t/s/u`
- ITL P50: `15.75 ms`, equivalent to `63.49 t/s/u`

The mean vLLM decode number includes vLLM request handling, host token
application, token/logprob readback, and first decode trace setup per request.
The warmed steady ITL P50 is within about 10% of the direct full-model
no-readback token-out latency. Scheduler overlap remains disabled until a
separate overlap proof exists.

## Profiler Evidence

No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER`, live-server profiling,
or `ttnn.ReadDeviceProfiler` was run for this vLLM stage. This follows the
vLLM-stage hardware-safety rule in the selected skills.

Device-op context for the serving decode path comes from the delegated
optimized full-model token-out trace, which includes final norm, LM head,
split sampling, and token feedback:

- `../optimized_full_model/reduced_profile_summary.json`
- `../optimized_full_model/tracy/reduced_profile/reduced_profile_ops_perf_results.csv`
- `../optimized_full_model/tracy/reduced_profile/reduced_token_out_decode_perf_report.txt`
- `../optimized_full_model/tracy/reduced_profile/reduced_token_out_decode_perf_report.csv`

That profile shows the reduced one-layer full token-out replay at
`1.55 ms` min / `1.58 ms` avg and the full 32-layer token-out path within about
1-2% of the stack-plus-terminal estimate.

## Artifacts

- `perf_summary.json`
- `vllm_async_contract_probe.py`
- `vllm_async_contract_probe.json`
- `vllm_async_contract_probe.log`
- `runtime_fallback_audit.txt`
- `artifacts/before_vllm_integration/`
- `artifacts/after_async_token_refresh_benchmark/`
- `artifacts/after_async_token_refresh_sampling_qualitative/`

## Limitations

- `max_num_seqs=1`.
- `tt_async_decode_allows_overlap=False`; async scheduler overlap is not
  enabled.
- Mean per-request decode still includes first decode trace setup for each new
  request. Keeping a decode trace live across prefill is not accepted without a
  focused proof because active traces and persistent decode buffers can make
  subsequent prefill allocations unsafe.
- Host-only sampling features and top-k/all-vocab logprob requests remain
  skipped or rejected under strict on-device sampling.
- The adapter contract probe passed but the process hit an ARC timeout during
  device close; bounded `tt-smi -r`, `tt-smi -ls --local`, and a mesh smoke
  recovered the T3K before serving benchmarks were run.
