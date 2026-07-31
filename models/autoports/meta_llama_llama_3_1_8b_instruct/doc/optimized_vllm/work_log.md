# Optimized vLLM Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`

## Starting Point

Started from the completed vLLM integration stage and the datatype-sweep
selected config:

- `doc/datatype_sweep/selected_precision_config.json`
- attention and MLP weights: `bfloat4_b`
- LM head: `bfloat8_b`
- activation, residual, CCL, KV cache, MLP multiply: `bfloat8_b`
- attention and LM head fidelity: `HiFi2`
- MLP fidelity: `LoFi`
- norms: BF16 / `HiFi4`

The vLLM integration before benchmark was archived before overwriting
`readiness_vllm`:

- `artifacts/before_vllm_integration/vllm_benchmark.json`
- `artifacts/before_vllm_integration/server.log`
- `artifacts/before_vllm_integration/sampling_tests.log`
- `artifacts/before_vllm_integration/vllm_qualitative_outputs.json`
- `artifacts/before_vllm_integration/check_degenerate_output.log`

## Code Changes

`tt/generator.py`:

- Added counters for nonblocking model trace replay and the one-time sync before
  first sampling trace capture.
- Changed model decode trace replay to always call
  `ttnn.execute_trace(..., blocking=False)`.

`tt/generator_vllm.py`:

- Added `_expected_next_decode_token`.
- After prefill/decode host formatting, record the sampled token returned by
  on-device sampling.
- In decode, copy the host token into the persistent trace input only when the
  scheduler token differs from the expected on-device feedback token, or when
  the request/trace state requires a reset.
- Reset the expected token at request/reset boundaries.

`doc/optimized_vllm/vllm_async_contract_probe.py`:

- Added focused adapter-contract coverage through the actual
  `Llama31_8B_InstructForCausalLM` class and a vLLM-owned KV cache.

## Static Validation

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator_vllm.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_vllm/vllm_async_contract_probe.py
```

Result: pass.

## Async/Stale-Input Probe

Command:

```bash
timeout 7200s python_env/bin/python -u \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_vllm/vllm_async_contract_probe.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-model-len 1024 \
  --block-size 64 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_vllm/vllm_async_contract_probe.json
```

Result JSON status: `pass`.

Passed assertions:

- async capability true and scheduler overlap false;
- `decode_forward(..., read_from_device=False)` returns device tensors;
- `read_decode_output(..., async_read=True)` records deferred events;
- all probe model trace replays are nonblocking;
- unchanged page table is not recopied;
- unchanged token feedback stays on device;
- changed host token copies once without position/page-table copy;
- changed current position refreshes position and RoPE;
- changed page table copies once and is marked changed-only;
- greedy path uses split sampling with `force_argmax=False`.

The probe process wrote passing JSON, then hit an ARC timeout while closing the
mesh:

```text
Timed out after waiting 1000 ms for ARC to respond
Read 0xffffffff over PCIe ID 0: the board should be reset
```

Recovery:

```bash
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
python_env/bin/python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: reset succeeded, all 8 devices listed, mesh smoke printed
`MESH_SMOKE_OK`.

## After Benchmark

Command:

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

Result: pass, server terminated cleanly.

Before metrics:

- TTFT P50/P99: `81.10 ms` / `1047.01 ms`
- ITL P50/P99: `15.77 ms` / `25.09 ms`
- Aggregate output throughput: `51.84 tok/s`
- Mean per-user decode: `53.90 t/s/u`

After metrics:

- TTFT P50/P99: `80.98 ms` / `990.80 ms`
- ITL P50/P99: `15.75 ms` / `22.71 ms`
- Aggregate output throughput: `52.07 tok/s`
- Mean per-user decode: `54.10 t/s/u`

Artifact:

- `artifacts/after_async_token_refresh_benchmark/vllm_benchmark.json`
- `artifacts/after_async_token_refresh_benchmark/server.log`

## Final Sampling And Qualitative

Command:

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

Result: pass, server terminated cleanly.

Sampling:

```text
41 passed, 22 skipped, 9 xfailed, 16 warnings in 120.71s
```

Qualitative verdict: coherent and on topic. Known minor leading-token
duplication remains (`a a`, `The The`), with no gibberish, collapse, or
cross-request contamination.

Degeneracy check:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --missing-artifacts critical \
  --scope vllm
```

Result:

```text
No degenerate output detected.
```

Artifacts:

- `artifacts/after_async_token_refresh_sampling_qualitative/sampling_tests.log`
- `artifacts/after_async_token_refresh_sampling_qualitative/server.log`
- `artifacts/after_async_token_refresh_sampling_qualitative/vllm_qualitative_outputs.json`
- `artifacts/after_async_token_refresh_sampling_qualitative/check_degenerate_output.log`

## Runtime And Fallback Audit

Artifact: `runtime_fallback_audit.txt`.

Source audit:

- vLLM plugin calls `decode_forward(..., read_from_device=False)`, then
  `read_decode_output(..., async_read=True)`, then
  `process_decode_output_host(...)`.
- The adapter rejects non-token host output processing and missing sampling
  params.
- The adapter has no host greedy/top-1 argmax, no full-logits decode readback,
  and no adapter-side fallback sampler.
- Greedy sampling remains the full-model split-sampling path with
  `force_argmax=False`.

Server logs:

- `sample_on_device_mode=all`
- `trace_mode=decode_only`
- `async_scheduling=True`
- no `Forcing argmax sampling` log lines found in the archived after logs.

Final process and device audit:

```bash
pgrep -af "vllm.entrypoints|EngineCore|TTEngineCore|run_vllm_server" || true
timeout 60 tt-smi -ls --local
```

Result: only the audit command matched; all 8 devices listed.

## Profiler Policy And Evidence

No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER`, live-server profiler,
or serving-adapter profiler collection was attempted for the vLLM stage.

Equivalent device-op context is the optimized full-model reduced token-out
profile because vLLM delegates to the same model/generator split-sampling path:

- `../optimized_full_model/reduced_profile_summary.json`
- `../optimized_full_model/tracy/reduced_profile/reduced_profile_ops_perf_results.csv`
- `../optimized_full_model/tracy/reduced_profile/reduced_token_out_decode_perf_report.txt`
- `../optimized_full_model/tracy/reduced_profile/reduced_token_out_decode_perf_report.csv`

## Optimize Checklist Closure

- Decoder path fully traced with no host fallbacks: covered by
  `vllm_async_contract_probe.json`, source audit, and `runtime_fallback_audit.txt`.
- Decode activations, prefill activations, matmul configs, memory configs,
  SDPA, DRAM-sharded decode matmuls, CCL strategy, and dtype/fidelity policy:
  inherited from the optimized full-model and datatype-sweep stages, with
  artifacts under `../optimized_full_model/`, `../optimized_multichip_decoder/`,
  and `../datatype_sweep/`.
- LM head and sampling: vLLM delegates to the full-model split LM-head and
  `SamplingGenerator` path. The optimized full-model artifacts include final
  norm, LM head, logits movement, split sampling, and token feedback in the
  token-out profile. `vllm_async_contract_probe.json` proves vLLM uses
  `tt_out_tok` device feedback and `force_argmax=False`.
- Runtime fallback audit: `runtime_fallback_audit.txt` records no host
  greedy/top-1 argmax, no full-logits decode readback, no eager sampling, and
  no leftover vLLM/EngineCore process in the measured path.
- vLLM before/after performance accounting: `perf_summary.json` records same
  workload TTFT, ITL, aggregate output throughput, mean per-user decode t/s/u,
  the optimized full-model comparison, and null device/profile fields with the
  no-live-vLLM-profiler reason.
- Watcher/profiler policy: no vLLM profiler collection was run. The final
  serving runs terminated cleanly, and final `tt-smi -ls --local` listed all
  8 devices.
- MoE routed-expert checklist item: not applicable to Llama 3.1 8B.

## Rejected Options

- Live vLLM profiler collection: rejected by selected skill guidance because
  T3K live-server profiling has a known hardware-stability risk.
- Force-argmax greedy: rejected. The generator enforces split sampling and the
  sampling strategy benchmark records `force_argmax` unavailable for this model
  contract.
- Adapter-side host argmax or full-logits readback: rejected because
  `sample_on_device_mode=all` must keep sampling on device.
- Scheduler overlap: rejected for now. `tt_async_decode_allows_overlap=False`
  because next-step scheduler state is host-owned and no overlap proof has been
  run.
- Keeping a decode trace live across prefill: not accepted without a separate
  proof. Existing optimized-full-model docs record same-process prefill after
  token-out trace as a known risk, and TT tracing guidance warns that allocating
  device buffers while an active trace exists can corrupt trace-owned buffers.

## Limitations

- `max_num_seqs=1`.
- Mean per-request vLLM decode remains below full-model no-readback steady
  replay because it includes host request handling, token/logprob readback, and
  first decode trace setup per request.
- The warmed steady ITL P50 (`15.75 ms`) is close to the optimized full-model
  no-readback steady latency (`14.17 ms`), but full serving mean is slower.
- Strict on-device sampling skips or rejects host-only request features and
  top-k/all-vocab logprobs for this model.
