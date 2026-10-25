# Qwen3.8-27B vLLM integration

**Primary: 445.55 ms TTFT; 36.24 tokens/s/user** at
**4,096 input / 128 output / batch 1 / one concurrent request**.
Full sampling passes (76 tests); qualitative review passes. Independent
[stage review](stage_review.md) returned **clean-pass**. Runner re-verification
returns **exit 1 (advisory)** for documented weight copies and KV-pool shortfall.
The reported critical context-cap finding was a synthetic host fixture, preserved
under `tests/evidence/`; the work log records the concrete refutation.
Local checkpoint SHAs are recorded in the work log; nothing was pushed.

| Workload and server capacity | TTFT P50 / P99 (ms) | TPOT mean / P50 / P99 (ms) | ITL P50 / P99 (ms) | Output tokens/s | 1000 / mean TPOT (tokens/s/user) |
|---|---:|---:|---:|---:|---:|
| **Primary 4096/128, 1 request, concurrency 1, max-num-seqs 1** | 445.546 / 445.546 | 27.591 / 27.591 / 27.591 | 27.616 / 28.496 | 32.405 | **36.244** |
| Secondary CI burst 100/100, 32 requests, no explicit concurrency cap (observed 32), max-num-seqs 32 | 2395.195 / 2395.813 | 112.060 / 112.047 / 112.359 | 109.197 / 204.297 | 237.215 | 8.924 (burst context only) |

Both use greedy temperature 0, ignore EOS, and complete all requested tokens.
The primary is one measured request, so its TTFT/TPOT P50 and P99 coincide; this
is not a tail-latency population estimate. CI admission can affect TPOT and is
not the headline decode result. Primary raw/normalized artifacts are
`readiness_vllm/vllm_result.json` / `vllm_benchmark.json`, with immutable
`final_b1_` copies. Secondary artifacts are `vllm_ci_serving_result.json` /
`vllm_ci_serving_benchmark.json`, with `full_b32_performance_` copies.

## Serving contract and sources

All 64 layers run on four Blackhole p300c devices (`P300x2`, mesh 1x4), with
`max_model_len=262144`, matching `doc/context_contract.json`. There is no advertised
context reduction. The selected `head_bfp8_hifi2` policy is loaded directly:
attention/MLP/down weights BFP4, embedding/norm BF16, head BFP8/HiFi2, projection
LoFi, KV BFP8, recurrent state FP32, and the selected activation/residual/CCL
policies and empty layer-exception map. Runtime manifests and preparation events
record the full policy.

Serving uses [tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin)
based on `1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf`, plus the upstream
[vLLM v0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) recommended by that
revision's `docs/install-vllm-tt.sh`/`docs/vllm-overrides.txt`. The empty-target
source build reports `0.30.0+empty`. [serving_stack.json](serving_stack.json)
records installation and registration provenance. Engine logs resolve
`TTQwen3_5ForConditionalGeneration` through `TT_MODEL_CLASS_OVERRIDES` to this
experiment's `tt/generator_vllm.py:Qwen38ForCausalLM`.

The adapter composes the existing generator and delegates prefill, continuation,
resident traced decode, and canonical split sampling. It does not implement a
performance sampler or read logits to feed tokens back through Python. Device
sampled tokens and positions remain resident across ordinary asynchronous steps;
only changed scheduler page tables are refreshed. Deferred output reads select
one replica and transfer token IDs for scheduler/API output. Synchronous output
formatting uses that same minimal read when given raw device output.

vLLM owns attention pages and per-layer recurrent snapshots. Exact allocated
cache objects pass into the generator. Full-attention prefix hits require valid
hashed KV blocks; linear-attention hits additionally require the matching
convolution/recurrent checkpoint at the aligned boundary. The shared hybrid-cache
manager uses the common valid prefix. Snapshot data moves on device, not through
host tensors. Standalone generator cache ownership remains separate.

## Correctness and trace evidence

- Full sampling profile: **76 passed, 74 skipped, zero failures**, through
  `run_vllm_server --sampling-profile full` at B8. Explicit
  `QWEN38_HOST_SAMPLING=1` and `seeded_sampling_policy=host` support shared tests
  requiring penalties, structured output or logits. These are compatibility
  results; both performance servers disable this mode. Of the skips, 71 concern
  unconfigured speculative decoding and three require an opt-in chunking budget
  not passed by the shared runner. See `sampling_verdict.json` and
  `full_b8_sampling_tests.log` under `readiness_vllm/`.
- Device-only seeded requests reproduce across three orderings without logits
  reads (`sampling_route_controls.json`). Full64/B32 standalone and serving
  reordered top-20 logits/tokens agree exactly (`standalone_route_logits.json`).
- Reduced adapter/generator tests verify stale-token/current-position behavior,
  delayed reads, slot remapping, exact state/RNG preservation, and changed versus
  unchanged page tables. Full64/B8 automatic allocator growth adds three distinct
  concurrent requests, 4,160 generated tokens each, crossing 4,096 and 8,192.
  All six asynchronous forward/reverse outputs equal the non-overlapped control
  token-for-token: 12 page-only refreshes, 8,304 unchanged-table steps, four
  captures, zero retirements and zero host-logits reads (`page_growth_async.json`).
- Final B32 mixed-shape lifecycle passes 112 requests on one server, with tracking
  enabled before TTNN import and program-cache allocations included. Logical
  lengths 137/138/139, 8,195/12,291 and awkward final chunks succeed. Actual
  scheduler starts 4,096/8,192/12,288 occur while another request decodes.
  Decode masked/unmasked and sampling/buffered-sampling signatures retain their
  four original trace IDs, with zero retirements/recaptures and zero logits reads.
  `full_b32_budget131072_lifecycle.json` maps events, signatures and raw requests.
- Multi-turn B32 traffic reuses 131,072 of 132,160 previous tokens (**99.1768%**)
  across 32 conversations of 4,130 tokens plus 33 appended tokens, including
  reversed admission. Cold/hit greedy outputs match. Per-layer hit rules, physical
  pool accounting and capture evidence are recorded in `doc/agentic_serving.json`.
- B16 independently passes 16 concurrent requests of 129 input/8 output tokens;
  all 2,064 input and 128 output tokens are accounted for. B1/B8/B16/B32 each have
  their own configuration in [batch_configurations.json](batch_configurations.json).

All twelve prompt-correct greedy/sampled qualitative outputs were read and judged
coherent and on-topic, without repetition loops, gibberish, unintended language
drift or visible request contamination. Matching full-model controls use the
checkpoint's chat template. Thinking text and story/thermodynamics truncation at
256 output tokens also appear in controls; these are not complete-answer claims.
The mechanical degeneracy gate passes. See `vllm_qualitative_verdict.json`,
`vllm_qualitative_outputs.json`, `full_model_controls.json` and
`full_model_control_verdict.json`. The independent reviewer also read outputs.

## Latency findings and limitations

The B32 scheduler uses `--max-num-batched-tokens 131072` and
`--long-prefill-token-threshold 4096`. On the same synthetic lifecycle, cold
32x4130-token turns improved from 524.390 to 33.763 seconds; prefill submissions
fell from 64 to four. Warm 32x4163-token turns take 3.446 seconds. Across all 112
mixed-shape requests, completion time is 255.429 seconds (0.43848 requests/s),
TTFT P50/P95/P99 is 4970.906/32686.050/32686.979 ms, and SSE pause P99/max is
425.512/31959.208 ms. These host-wall diagnostic measurements include first
encounters and are separate from warmed benchmark performance.

Long B32 prefills can still pause decode for about 32 seconds. Fixed-batch model
prefill compute remains visible after serial admission overhead is removed.
[The before/after analysis](b32_host_latency_analysis.md) and timeline retain both
runs. No device profiler, Tracy or perf-report was used in this serving stage.
Device-time and roofline telemetry therefore remain unknown.

B32 uses 1,024-token physical prefill pieces after a real larger-bucket allocation
failure; B1/B8/B16 use 4,096. This changes internal work size, not logical context.
B32 preparation checkpoints show about 29.31 GB allocated and 4.73 GB free per
device, excluding the separate 128 MiB trace reservation. These are allocator
checkpoints, not peak-memory measurements. The configured 289-ID hybrid pool
cannot retain all 32x32768 planning prefixes simultaneously; that larger workload
was not measured. This is a configured-layout shortfall, not a claim that all
possible layouts face a hard physical limit or a reduced advertised context.

The Stage08 teacher-forcing result (203 input/100 output/B1, 36.381 tokens/s/user)
is only a decoder latency lower-bound reference and is not a serving result or a
like-for-like overhead comparison. The matching Stage08 autoregressive
4096/128/B1 result is 36.171 tokens/s/user. Production serving uses nonblocking
trace replay, resident token/position state, changed-only page updates, and no
logits readback. Duplicate per-layer host page-table padding was removed; remaining
parameter marshalling measured 3-6 microseconds per step in a host-only probe.
[Runtime audit](runtime_audit.md) records the evidence and limits of that audit.

## Reproduction and artifacts

Run inside container `qwen38-functional-build`, from `/workspace`; checkpoint is
`/workspace/.cache/qwen38-full`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The wrapper sets mesh/runtime/plugin
registration and removes watcher/profiler settings. All runs use decode tracing,
128 MiB trace region and `FABRIC_1D_RING`. Primary command:

```sh
TT_METAL_TRACE_ALLOC_TRACKING=0 bash models/autoports/qwen_qwen3_8_27b/tests/serve_vllm.sh \
  --stages serve,benchmark --max-num-seqs 1 --no-benchmark-ci-serving
```

The wrapper explicitly requests 4,096 benchmark input tokens; output defaults to
128, one request and concurrency one. For the secondary B32 run, use
`--max-num-seqs 32` and
`--additional-server-args='--language-model-only --mamba-cache-mode align --mamba-block-size 4096 --max-num-batched-tokens 131072 --long-prefill-token-threshold 4096 --max-logprobs -1 --async-scheduling'`.
The runner also executes CI100/100/32 by default. Host compatibility and event
logging must remain unset for performance. Exact argv, relevant environment and
source hashes are in `final_b1_command.json` and `full_b32_performance_command.json`.

The [evidence index](evidence_index.md) maps immutable raw/normalized pairs and
controls; [work_log.md](work_log.md) records commands, failures, repairs and commits.
A client event-loop teardown warning appears in an otherwise passing sampling
test. Upstream abort shutdown can force-terminate EngineCore; final process and
open-device-descriptor audits establish cleanup, not graceful release in every
run. No device reset was required. See `final_cleanup.json` for final ownership.
