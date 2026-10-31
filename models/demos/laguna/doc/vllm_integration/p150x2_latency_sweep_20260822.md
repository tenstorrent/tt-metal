<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 p150x2 production latency sweep (2026-08-22)

Result: **PASS, 9/9 requests**. Every row completed with exactly 512 output tokens at concurrency 1;
there were no failed or aborted requests, and final health was HTTP 200.

Hardware: two P150 Blackhole ASICs in a 1x2 mesh on one physical dual-P150 card. The production
`p150x2` profile served a 131,072-token context with streaming prefill and qualified prefix caching
enabled. Each row used a unique cache salt, so this is a genuinely cold curve despite the production
cache policy.

| Requested ISL | Actual prompt | Actual total | OSL | C | TTFT | TPOT | E2EL | Decode tok/s/user | Aggregate output tok/s | TTFT speedup | E2EL speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 594 | 512 | 1 | 0.215 s | 50.07 ms | 25.801 s | 19.97 | 19.844 | 0.959x | 1.000x |
| 1,024 | 1,066 | 1,578 | 512 | 1 | 2.213 s | 50.45 ms | 27.995 s | 19.82 | 18.289 | 1.001x | 1.000x |
| 2,048 | 1,939 | 2,451 | 512 | 1 | 2.563 s | 50.49 ms | 28.366 s | 19.80 | 18.050 | 1.002x | 1.000x |
| 4,096 | 4,138 | 4,650 | 512 | 1 | 8.984 s | 50.56 ms | 34.822 s | 19.78 | 14.703 | 1.001x | 1.000x |
| 8,192 | 8,234 | 8,746 | 512 | 1 | 19.680 s | 50.71 ms | 45.592 s | 19.72 | 11.230 | 1.002x | 1.001x |
| 16,384 | 16,426 | 16,938 | 512 | 1 | 33.931 s | 51.00 ms | 59.993 s | 19.61 | 8.534 | **1.377x** | **1.213x** |
| 32,768 | 32,810 | 33,322 | 512 | 1 | 67.812 s | 51.60 ms | 94.178 s | 19.38 | 5.437 | **1.801x** | **1.577x** |
| 65,536 | 65,578 | 66,090 | 512 | 1 | 156.630 s | 52.77 ms | 183.595 s | 18.95 | 2.789 | **2.292x** | **2.102x** |
| 130,048 | 130,090 | 130,602 | 512 | 1 | 380.812 s | 55.10 ms | 408.967 s | 18.15 | 1.252 | 1.002x | 1.002x |

Speedup is the 2026-08-21 historical latency divided by the current latency, so values above 1 are
faster. The actual prompt lengths match the historical run exactly. Streaming removes the former
power-of-two over-padding cliff: cold TTFT improves by 1.377x at 16K, 1.801x at 32K, and 2.292x at
65K. The 130K request is already near the aligned 131,072-row cap, so both implementations compute the
same total rows and its latency is essentially unchanged. Through 8K, E2EL differs by at most 0.101%;
across all nine points, TPOT is unchanged to within 0.035% and is slightly lower in every row.

## Method and gates

- `vllm bench serve`, OpenAI chat endpoint, random requested ISL, OSL 512, concurrency 1, request rate
  infinity, temperature 0, ignore EOS, and seed 1234. Each point is one sample, not a variance study.
- `--ready-check-timeout-sec 0` and `--num-warmups 0` disabled benchmark-generated probes and warmups.
  Server boot had already warmed the finite streaming ladder and canonical long-stream case, then
  captured decode trace and froze 826 TTNN program-cache entries before measurement.
- Every request supplied `cache_salt=laguna-final-sweep-20260822-isl<requested_isl>`. Final Prometheus
  values were `prefix_cache_queries_total=260363`, the sum of the nine actual prompt lengths,
  `prefix_cache_hits_total=0`, and `prompt_tokens_cached_total=0`.
- The bounded measurement-log segment, bytes `[34748,49193)`, contains exactly nine tokenize 200s and
  nine chat-completion 200s. Request-success and prompt/generation histogram counts are each nine;
  abort and error counts are zero. It has no prefix resume, serving-time program-cache miss or compile,
  traceback, fatal, hang, watcher, OOM, or device-death marker. The only broad fault-scan match is the
  known once-only active-trace allocator advisory at the first eager prefill; it reports a theoretical
  risk, not an observed corruption. Final `/health` returned 200.
- Requested ISL and actual server-counted prompt tokens are both reported. Random token IDs were
  decoded to text, wrapped in the chat template, and tokenized again, so those lengths need not match.
- `Decode tok/s/user = 1000 / mean_tpot_ms`. With one request, `E2EL = duration` and aggregate output
  throughput is `512 / duration`; both include cold-prefill time.

Raw results were written to `/tmp/laguna-final-sweep-20260822/results/isl*-cold-c1.json`; the bounded
server log is `/tmp/laguna-final-sweep-20260822/laguna_serve_20260822-125448.log`. Full-precision values
are committed in [`p150x2_latency_sweep_20260822.tsv`](p150x2_latency_sweep_20260822.tsv). The
pre-streaming, cache-off baseline and its method remain in
[`p150x2_latency_sweep_20260821.md`](p150x2_latency_sweep_20260821.md) and
[`p150x2_latency_sweep_20260821.tsv`](p150x2_latency_sweep_20260821.tsv).
