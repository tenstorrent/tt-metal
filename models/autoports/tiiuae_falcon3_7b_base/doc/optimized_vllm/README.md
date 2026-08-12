# Falcon3-7B-Base optimized vLLM serving

## Result

The final real vLLM TT-plugin path through `tt/generator_vllm.py` serves a
128-token prompt and generates 128 tokens for one request at **183.68 ms TTFT**
and **62.36 TPOT-derived decode tokens/s/user** (16.037 ms mean TPOT).  Median
ITL is 14.573 ms (68.62 tokens/s/user), p99 ITL is 14.832 ms, and aggregate
output throughput is 57.64 tokens/s.  This is the primary headline workload:
batch/concurrency 1, greedy temperature 0, explicit generation config,
`ignore_eos`, `max-num-seqs=32`, `max-model-len=32768`, TP4 1x4 P300 mesh, and
the selected `all_bfp4_lofi_bf16_kv` policy.

The final path preserves the 32,768-token context contract and valid
non-aligned prompts.  A 37-token vLLM prompt completed successfully after the
change.  It uses the canonical full-model split greedy sampler on device; it
does not read full logits, run host argmax, or use eager sampling.

## Same-runner before and after

Both rows below used the exact workload and server configuration stated above.

| 128 prompt / 128 output / 1 request | Before | Final |
|---|---:|---:|
| TTFT p50/p99 (ms) | 182.47 / 182.47 | 183.68 / 183.68 |
| mean TPOT (ms) | 15.888 | 16.037 |
| ITL p50/p99 (ms) | 14.566 / 15.121 | 14.573 / 14.832 |
| TPOT-derived decode (tokens/s/user) | 62.94 | 62.36 |
| aggregate output throughput (tokens/s) | 58.17 | 57.64 |

The result is statistically flat (headline decode -0.9%), so no speedup is
claimed.  The retained work tightens the asynchronous plugin boundary: steady
decode reuses the persistent device page table instead of comparing the full
fixed-width host table every token, and deferred output reads submit only the
first of four identical sampled-token replicas.  A synchronous-read control
regressed to 57.5 tokens/s/user and was rejected.  Reusing Python scheduler
payload objects was also neutral and was removed from the external vLLM tree.

## Full-model comparison

The comparable non-serving harness uses the serving physical contract:
maximum batch 32 with one active slot, token `[1,1,1,32]`, positions `[32]`,
page table `[32,1024]`, residual `[1,1,32,3072]`, 4,128 external cache blocks,
32,768 context, and the same model and sampling traces.  It measured 14.732 ms
caller-visible per token, or 67.88 tokens/s/user.  Final vLLM median ITL is
14.573 ms, or 68.62 tokens/s/user, so serving decode is about as fast as the
optimized full-model path for comparable physical work.  The earlier
110.38 tokens/s physical-batch-1 number is not a like-for-like comparison.

## CI serving-burst evidence

This secondary capacity/nightly-parity workload is 100 prompt tokens, 100
output tokens, 32 requests submitted without an explicit concurrency cap,
greedy temperature 0, and otherwise the same server, mesh, context, and TT
configuration.  It is not the headline per-user decode result.

| 100 prompt / 100 output / 32 requests | Before | Final |
|---|---:|---:|
| TTFT p50/p99 (ms) | 414.44 / 415.58 | 415.41 / 416.63 |
| mean / p99 TPOT (ms) | 16.860 / 18.348 | 16.876 / 18.371 |
| ITL p50/p99 (ms) | 15.071 / 74.718 | 15.056 / 76.059 |
| aggregate output throughput (tokens/s) | 1539.74 | 1537.89 |
| TPOT-derived rate (tokens/s/user, secondary) | 59.31 | 59.26 |

## Correctness and implementation evidence

- Full vLLM sampling suite: 72 passed, 1 skipped (unsupported beam search), no
  failures.  Greedy and sampled requests use `sample_on_device_mode=all`.
- Shared qualitative suite: coherent, on-topic base-model continuations; the
  visible continuation into adjacent Q&A material also occurs in exact HF
  controls and is expected base-model behavior.  Prompt formatting metadata and
  HF controls remain in `readiness_vllm/`.
- Degenerate-output checker: pass.  Non-aligned 37-token serving: pass.
- Contract tests: 7 passed, including changed token/current-position handling,
  changed page tables on scheduler reset, unchanged page-table reuse, rejection
  of live slot remapping, and one-replica deferred reads.
- `supports_async_decode` is enabled only for the TT vLLM async scheduling path.
  Steady `decode_forward(..., read_from_device=False)` returns a device tensor;
  model and sampling traces replay with `ttnn.execute_trace(..., blocking=False)`.
  The plugin defers the single token-shard host read and waits only at its
  output-consumption boundary.
- AutoFix phase evidence observed 224 steady async steps, maximum pending depth
  2, and only two forced drains at scheduler layout boundaries.  Median queue
  time before the deferred wait was 13.736 ms, demonstrating that device work
  crosses the adapter's asynchronous boundary.
- Token, current-position, RoPE, page-table, cache, sampler parameters, and
  sampled-token feedback are persistent device tensors.  Tokens/positions are
  refreshed at scheduler reset; the page table is refreshed only when the
  scheduler state changes.
- Runtime fallback audit found no host greedy/top-1 path, full-logits readback,
  eager sampler, standalone-cache assumption, or newly introduced
  torch/from_torch/to_torch, tilize/untilize, reshard, or blocking-read path.

No Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or
`ReadDeviceProfiler` was collected.  This is intentional for the vLLM-serving
stage; benchmark JSON, sampling/qualitative tests, stale-input checks, async
phase evidence, and source-level no-host-fallback evidence are used instead.

## Artifacts

- `results/before/` and `results/after/`: same-runner primary and CI JSON/logs
- `results/autofix/full_model_batch32_active1.json`: comparable full-model run
- `results/autofix/async_phase_trace.3520406.json`: async-boundary counters
- `results/autofix/sync_read_control_benchmark.json`: rejected sync control
- `results/after/sampling_tests.log`: final sampling suite
- `results/after/non_aligned_prompt_37.json`: final non-aligned request
- `results/after/vllm_qualitative_outputs.json`: final generated text
- `perf_summary.json`, `work_log.md`, `AUTOFIX.md`: compact stage accounting

## Limitations

Mean TPOT includes request-boundary timing and is lower than median steady ITL;
both are reported.  Full-context batch 1 and serving batch 32 are validated,
but full-context batch 32 is not advertised.  The retained boundary cleanup is
performance-neutral in this workload.
