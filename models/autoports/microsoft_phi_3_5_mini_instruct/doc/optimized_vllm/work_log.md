# Optimized vLLM Work Log

Date: 2026-06-15

## Starting Point

Started from completed vLLM integration for `microsoft/Phi-3.5-mini-instruct` and selected datatype config `doc/datatype_sweep/selected_precision_config.json` (`c004_default_weights_bf16_ccl`). The serving adapter is `models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator_vllm.py`.

Baseline same-harness serving command:

```bash
VLLM_TT_SKIP_HOST_ONLY_SAMPLING_TESTS=1 python -m models.common.readiness_check.run_vllm_server \
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

Baseline artifact copied to `readiness_vllm/vllm_benchmark_before_optimized_vllm.json`: TTFT P50/P99/mean `170.79/222.99/173.47 ms`, ITL P50/P99/mean `19.33/21.11/19.45 ms`, aggregate output throughput `48.38 tok/s`, mean per-user decode `51.41 t/s/u`.

## Implemented Optimized Serving Contract

Generator changes in `tt/generator.py`:

- Added token-out serving trace state for the sampled device output and sampling-mode key.
- vLLM decode now calls `_ensure_decode_trace(..., capture_sampling=True)`.
- The serving trace captures model decode, LM head, and `SamplingGenerator.sample(..., tt_out_tok=self.trace.token_input)` in one replayable token-out trace.
- Hot serving replay calls `ttnn.execute_trace(..., blocking=False)` and returns `self.trace.sampled` when `read_from_device=False`.
- Sampling mode changes recapture the serving trace so greedy/logprob/penalty state is not reused across incompatible modes.
- Page-table refresh compares host tables; unchanged tables reuse the persistent device tensor, changed tables update it in place.
- Prefill token/logit/sample temporaries are explicitly deallocated after host readback.
- Standalone optimized full-model generation keeps the split-sampling trace path and still uses the full-model `SamplingGenerator` contract.

Adapter contract in `tt/generator_vllm.py`:

- `supports_async_decode=True`
- `tt_async_decode_allows_overlap=False`
- `supports_prefix_caching=False`
- `supports_sample_on_device=True`
- generator initialization uses `allocate_standalone_cache=False`;
- vLLM-owned `kv_cache`, `page_table`, token, and current-position inputs are passed by identity;
- `decode_forward(..., read_from_device=False)` delegates the split-read flag and returns device output.
- `read_decode_output(..., async_read=True)` delegates the deferred read.
- `process_decode_output_host(...)` delegates host formatting only.
- Missing `sampling_params` raises `ValueError`, so host sampling fallback is rejected.

## Rejected Options

- Force argmax: rejected because serving must reuse the full-model split-sampling/token-feedback contract and support top-k/top-p/logprob modes without host fallback.
- Full-vocab all-gather or host top-1 argmax: rejected because it violates `sample_on_device_mode=all` and the measured path must not read full logits.
- Separate second sampler trace for vLLM hot decode: rejected for serving because capturing the sampler inside the token-out serving trace avoids active-trace allocation hazards while preserving `SamplingGenerator` semantics and `tt_out_tok` feedback.
- Releasing the decode trace at every prefill: tested and rejected. It removed the allocator warning but degraded the same benchmark to `41.56 t/s/u`, aggregate `39.74 tok/s`, ITL mean `24.06 ms`.
- Live vLLM `tt-perf-report`, Tracy, `TT_METAL_DEVICE_PROFILER`, or `ttnn.ReadDeviceProfiler`: rejected by selected skill guidance for T3K vLLM serving stages.
- Async scheduler overlap: left disabled because vLLM constructs next-step request state from host scheduler inputs; no overlap proof was produced.

## Hardware Recovery Note

An earlier reduced trace smoke wrote `readiness_vllm/optimized_vllm_trace_variant_smoke.json` but then failed during mesh close with:

```text
Timed out after waiting 1000 ms for ARC to respond.
Message code 0xaa34 with arguments 0xffff and 0xffff
```

The stage was later resumed from existing artifacts. T3K reset recovery was performed with bounded `tt-smi` reset/list commands, followed by a 2x4 mesh open/close smoke that returned `MESH_SMOKE_OK`. The final full `run_vllm_server` run then completed cleanly.

## Final vLLM Run

Command:

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

Runner output:

- Server ready after about 160 seconds.
- vLLM TT plugin sampling tests: `59 passed, 13 skipped`.
- Qualitative prompts: 6 greedy and 6 sampled completions saved.
- Benchmark: 32 requests completed in 83.7 seconds.
- Server terminated cleanly.

Benchmark artifact: `readiness_vllm/vllm_benchmark.json`.

```json
{
  "ttft_ms": {"p50": 173.13391699281055, "p99": 284.6028610074427, "mean": 177.19703628245043},
  "itl_ms": {"p50": 19.39126499928534, "p99": 20.60809900285676, "mean": 19.439656400498105},
  "output_throughput_tok_per_s": 48.33464860300155,
  "mean_per_request_decode_tps": 51.438285865596775
}
```

## Before And After

| Metric | Before | After |
| --- | ---: | ---: |
| Completed requests | 32 | 32 |
| Prompt/output length | 128 / 128 | 128 / 128 |
| Concurrency | 1 | 1 |
| TTFT P50 / P99 / mean | 170.79 / 222.99 / 173.47 ms | 173.13 / 284.60 / 177.20 ms |
| ITL P50 / P99 / mean | 19.33 / 21.11 / 19.45 ms | 19.39 / 20.61 / 19.44 ms |
| Aggregate output throughput | 48.38 tok/s | 48.33 tok/s |
| Mean per-user decode | 51.41 t/s/u | 51.44 t/s/u |

The after result preserves throughput while moving the measured path onto the optimized serving contract.

Full-model comparison:

- Selected datatype token-out no-readback: `56.3713 t/s/u`.
- Optimized full-model token-out no-readback: `56.4306 t/s/u`.
- Optimized vLLM serving decode: `51.4383 t/s/u`, about `91.2%` of optimized full-model token-out.

## Closure Checks

Degenerate output:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --missing-artifacts critical \
  --scope vllm
```

Result: `No degenerate output detected.`

Optimized-vLLM contract:

```bash
python models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_vllm/contract_check.py
```

Result: pass, wrote `readiness_vllm/optimized_vllm_contract_checks.json`.

Adapter contract:

```bash
python models/autoports/microsoft_phi_3_5_mini_instruct/doc/vllm_integration/adapter_contract_check.py
```

Result: pass, wrote `readiness_vllm/adapter_contract_checks.json`.

Compile:

```bash
python -m compileall -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator_vllm.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_vllm/contract_check.py
```

Result: pass.

Log audit:

- `Allocating device buffers is unsafe`: exactly one match in `readiness_vllm/server.log`.
- Forbidden hot-path signatures: no matches for `Traceback`, runtime `ERROR`, `Exception`, host argmax/greedy, force-argmax, full-logits readback, `TopKDeviceOperation`, `ArgMaxDeviceOperation`, `blocking=True`, `ReadDeviceProfiler`, `TT_METAL_DEVICE_PROFILER`, `tt-perf-report`, or `tracy`.
- `fallback` matches only config text (`throw_exception_on_fallback=false` and vLLM structured-output config).

Process cleanup:

- Final narrowed `pgrep -af 'vllm.entrypoints.openai.api_server|EngineCore_DP|models.common.readiness_check.run_vllm_server|TT_METAL_DEVICE_PROFILER|tt-smi|tracy'` returned no leftover serving process after checks completed.

Stale-input evidence:

- `adapter_contract_checks.json` verifies token, current-position, page-table, and KV-cache identity delegation through the adapter.
- `optimized_vllm_contract_checks.json` verifies unchanged-page-table no-copy and changed-page-table in-place copy branches.
- `optimized_vllm_trace_variant_smoke.json` verifies nonblocking no-readback replay, persistent sampled-token feedback, and no full-logits decode readback.

## Artifacts

- `readiness_vllm/server.log`
- `readiness_vllm/sampling_tests.log`
- `readiness_vllm/vllm_qualitative_outputs.json`
- `readiness_vllm/vllm_benchmark_before_optimized_vllm.json`
- `readiness_vllm/vllm_benchmark.json`
- `readiness_vllm/adapter_contract_checks.json`
- `readiness_vllm/optimized_vllm_contract_checks.json`
- `readiness_vllm/optimized_vllm_trace_variant_smoke.json`
- `doc/optimized_vllm/perf_summary.json`

## Limitations

- Batch-1 serving only (`max_num_seqs=1`, `tt_data_parallel=1`).
- Prefix caching remains disabled.
- Async scheduler overlap remains disabled.
- One request-boundary active-trace allocator warning remains; it is documented and not a hot decode fallback.
- No live vLLM profiler was collected. For vLLM stages the selected skill requires same-harness serving metrics and contract checks instead; device-op context comes from optimized full-model reduced profiler artifacts.
