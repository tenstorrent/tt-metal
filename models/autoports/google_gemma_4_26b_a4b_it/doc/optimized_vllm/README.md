# Optimized Gemma 4 26B A4B vLLM serving

**Warmed primary single-user vLLM TT serving, prompt 128 / output 128 / one
request / concurrency 1: TTFT P50/P99 201.59/201.59 ms and decode 28.04 t/s/u
from mean TPOT 35.67 ms.** ITL P50/P99 is 33.91/34.69 ms and aggregate output
throughput is 27.05 tok/s. This is real vLLM TT-plugin execution through
`tt/generator_vllm.py`, with `sample_on_device_mode=all`, async scheduling,
nonblocking model and sampling trace replay, and the selected datatype policy.

## Before and after

Both runs used the same `run_vllm_server` configuration: P300x2 (one 1x4 P300C
mesh), `max_num_seqs=32`, `max_model_len=262144`, greedy temperature 0,
`trace_region_size=220000000`, and `FABRIC_1D_RING`.

| Primary workload (128 input / 128 output / 1 request / concurrency 1) | Before | After |
|---|---:|---:|
| TTFT P50 / P99 | 214.35 / 214.35 ms | 201.59 / 201.59 ms |
| mean / P99 TPOT | 46.48 / 46.48 ms | 35.67 / 35.67 ms |
| ITL P50 / P99 | 44.07 / 49.88 ms | 33.91 / 34.69 ms |
| aggregate output throughput | 20.92 tok/s | 27.05 tok/s |
| TPOT-derived decode | 21.51 t/s/u | 28.04 t/s/u |

Decode improved 30.3% and TTFT improved 6.0%. Each authoritative measurement
was the second identical `run_vllm_server` benchmark against one held server;
the first 128/128/1 run explicitly warmed the full request path. First-measured
artifacts are retained separately: the optimized first request exposed sampler
materialization in TTFT, while the detached baseline also compiled an uncached
kernel, so neither is blended into the warmed comparison.

| Secondary CI serving burst (100 input / 100 output / 32 requests / unconstrained admission) | Before | After |
|---|---:|---:|
| completed | 32/32 | 32/32 |
| TTFT P50 / P99 | 3949.91 / 5956.38 ms | 4270.71 / 6178.79 ms |
| mean / P99 TPOT | 418.91 / 450.69 ms | 401.61 / 435.28 ms |
| ITL P50 / P99 | 369.16 / 1520.83 ms | 359.07 / 1294.92 ms |
| aggregate output throughput | 69.57 tok/s | 71.81 tok/s |

The burst is capacity/nightly-parity evidence, not headline decode t/s/u.

## Optimization and trace contract

The old greedy path gathered the full vocabulary and forced device argmax. Its
semantic split alternative used one slow TopK over each 65,536-wide local shard.
The selected path keeps logits TP4-sharded, splits every local shard into two
32,768-wide multi-core TopKs, merges the 64 local candidates to 32, gathers only
candidate values/indices, then executes semantic `k=1, p=0, temp=1` sampling.
It writes the chosen token directly to the persistent `tt_out_tok` consumed by
the next model replay. `allow_force_argmax=False`; the measured server log has
no force-argmax marker, host argmax, or full-logits readback.

`decode_forward(..., read_from_device=False)` returns the device token tensor.
Both model and sampler traces replay with `ttnn.execute_trace(...,
blocking=False)`. `read_decode_output(async_read=True)` starts only the minimal
token read and records the fence; host formatting occurs afterward in
`process_decode_output_host`. Token, current-position, RoPE position, per-layer
page tables, KV cache, sampler parameters, and sampler output are stable device
tensors. Device token feedback and position advance happen once per emitted
token; unchanged page tables are not copied.

The async-overlap test passed with isolated and staggered outputs byte-identical,
a 96-token decode crossing the 64-token page boundary, an independent request
ending after three tokens, and no doubled-token or repeated-phrase failure. A
separate focused TP4 hardware probe deliberately supplied stale host token `0`
and position `999`: token/position refreshes stayed zero, device positions moved
33→34→35, the initial scheduler-table bind counted one copy, an unchanged
replay stayed at one, one mutated mapping advanced it exactly once to two, and
its next reuse stayed at two. Stable device addresses did not change and device
contents exactly matched both mutated scheduler tables. The sampled output and
next-token input had the same persistent device address. The adapter's async
read returned one event, which was synchronized before host token formatting.

## Correctness and capability

- Full vLLM plugin sampling profile: 72 passed, 1 skipped in 735.89 seconds.
- Six chat-templated greedy and sampled qualitative prompts are coherent and
  on-topic; the six-row comparison against the existing prompt-correct HF and
  optimized-full-model controls passes, and the mechanical degeneracy checker
  is clean.
- Existing integration evidence retains direct 47-token and 2051-token serving
  requests and a 262,143-input/1-output request. The context contract remains
  262,144; no prompt, benchmark, evaluation, or cache capability was reduced.
- Primary optimized vLLM decode is 28.04 t/s/u. Comparable optimized full-model
  host-visible decode is 26.21 t/s/u and no-host-boundary token-out is 28.02
  t/s/u (128 prompt / 128 generated / batch 1). Serving is within 0.1% of the
  latter and 7.0% faster than the host-visible generator loop.
- Final runs used `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`.
  No live vLLM or EngineCore process remained after shutdown; all four devices
  were visible.

## Optimize checklist

- Real shapes, selected precision, full 262,144 context, non-aligned prompts,
  and batch/concurrency 32 are preserved.
- Serving overhead was addressed before decoder math: force-argmax/full-vocab
  gather was replaced at the generator LM-head/sampler boundary.
- Persistent split traces, nonblocking replay, on-device token feedback,
  changed-only adapter page-table refresh, minimal async readback, and explicit host
  compatibility branches were audited.
- No Tracy, `tt-perf-report`, device profiler, adapter profiler, or
  `ReadDeviceProfiler` was collected in this serving stage.
- The final default was rerun through the same primary and CI benchmark harness;
  sampling, qualitative, async-overlap, fallback, and cleanup gates passed.

## Limitations

Prefix caching remains disabled. Top-k above 32 and features requiring full
host logits use the explicit compatibility mode and are excluded from measured
performance. P300C firmware 19.8.0 repeatedly failed to reactivate Ethernet core
29-25 when immediately reopening a mesh after a clean vLLM shutdown; bounded
`tt-smi` resets restored all four chips and subsequent mesh/server checks passed.
This is recorded as recoverable infrastructure behavior, not model evidence.

Artifacts are in `before/`, `after/`, and the model's `readiness_vllm/` folder.
