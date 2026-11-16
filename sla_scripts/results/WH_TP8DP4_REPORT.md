# Llama-3.3-70B, TP8/DP4 on one Wormhole Galaxy: results vs customer SLA

Code: tt-metal main f0a85c864bf, `models/ttt_compat` (4 x 1x8 submeshes).
Best config: reduce-scatter on 4 links + matmul configs (wo, w1w3, qkv). No precision change.

## Verdict

- One WH Galaxy with TP8/DP4 does not meet the SLA.
- It holds 128 users and meets the P99 decode floor (15), but only just at 4K context.
- It misses mean decode speed (25), long-prompt TTFT, and cold TTFT.

## SLA targets vs measured

| SLA item | Target | Measured (WH, TP8/DP4) | Result |
|---|---|---|---|
| Decode, mean concurrency (32 users) | 25 tok/s/user | 19.6 short context (best); 16.9 at 4K (stock) | Fail |
| Decode, P99 concurrency (128 users) | 15 tok/s/user | 19.6 short context; 15.45 at 4K (best) | Pass at 4K, no margin |
| Concurrency on one Galaxy | 128 | 128 users x ~4.2K context ran | Pass at 4K |
| TTFT, mean | 1 s | 0.76 s at 4K, single request, no load | Not measured under load |
| TTFT, P99 / cold at 16K | <= 5 s | 7.1 s, single request, no load | Fail |
| Cold TTFT at 32K | <= 10 s | 21.2 s, single request, no load | Fail |
| Max input 32K | runs | runs (16.3 tok/s/user decode) | Pass |
| Prefix cache, 40% hit on 2K prefix | supported | supported in code; not measured | Open |
| Accuracy, MMLU within 1 pt of BF16 | MMLU | top-1 97.0% / top-5 100% vs BF16 reference (baseline 96.8%); MMLU not run | Proxy pass; MMLU open |
| Peak arrival 240 RPM | 4 req/s | not measured; 4K prefill alone is ~0.76 s per group | At risk |

TTFT and decode numbers are from the tt-metal demo, not a vLLM server under load.

## Decode at 128 users, by context (best config unless noted)

| Prompt length | Users | ms/token | Tok/s/user | Total tok/s |
|---|---|---|---|---|
| ~128 tokens | 128 | 50.91 | 19.64 | 2,514 |
| ~4K tokens | 128 | 64.74 | 15.45 | 1,977 |
| ~128 tokens (stock) | 32 | 53.70 | 18.62 | 596 |
| ~4K tokens (stock) | 32 | 59.07 | 16.93 | 542 |

## Optimization steps (128 users, short context, accuracy-gated)

Gate: top-1 >= 96.4%, top-5 >= 99.8% vs BF16 reference (500 tokens, teacher-forced),
all 128 users produce output, no degenerate output. Full table: `TP8_STEPS.md`.

| Step | Tok/s/user | Kept? |
|---|---|---|
| Baseline (stock) | 18.36 | — |
| + Reduce-scatter on 4 links (was 1) | 19.32 | Yes |
| + Matmul configs (wo, w1w3, qkv) | 19.64 | Yes, best (+7%) |
| Norm all-gather links / 8B CCL config | 19.74 | No gain (noise) |
| Fused all-gather + matmul | 19.32-19.73 | No gain |
| SiLU fused into w1 matmul | 19.60 | No gain |

## Where the time goes (one layer, best config, 607 us device time)

| Part | us | Note |
|---|---|---|
| Matmuls | 372 | DRAM floor ~292 us |
| Reduce-scatter x2 | 64 | was 102 before the 4-link fix |
| All-gather x2 | 76 | fixed latency; link count has no effect |
| Norm, multiply, SDPA, other | 95 | |

## What it would take

- Config tuning is done. Remaining ideas are multi-day kernel or fusion work.
- Realistic ceiling: ~21-22 tok/s/user at short context; lower at 4K.
- 25 tok/s/user at 32 users is not reachable on WH with TP8/DP4.
- Options: TP32 on a WH Quad. BH Galaxy TP8/DP4 also misses: 23 tok/s/user for one user at 4K, 16.0 / 13.1 at 60 RPM (vLLM).

## TP32 vLLM load test

See `WH_TP32_VLLM_REPORT.md`. Short version: a WH Quad (TP32, 32 users each) is an estimate only. It is near pass at mean load and fails at peak, because TP32 prefill is not chunked yet.

## TP8/DP4 under vLLM traffic (WH, vLLM 0.31 plugin, DP4 x 32 seqs)

Results: `bench_wh_tp8dp4_mbt8192/`. Prefill budget 8192 tokens per step.

| Case | TTFT mean / P99 | Tok/s/user mean / P99 | SLA |
|---|---|---|---|
| Single 4K | 1.57 / 1.58 s | 16.6 | TTFT 1 s |
| Cold 16K single | 8.96 / 9.57 s | 11.6 | <= 5 s |
| Cold 32K single | 23.7 / 24.0 s | 8.3 | <= 10 s |
| 60 RPM | 3.38 / 7.68 s | 7.2 / 4.3 | 1 / 5 s; 25 / 15 |
| 60 RPM, 40% prefix | 2.81 / 6.59 s | 9.1 / 6.5 | same |
| 240 RPM | 16.6 / 48.9 s | 5.7 / 3.1 | same |

- Fails every SLA line under traffic.
- Decode under load is ~1/3 of the idle demo.
- DP engine 2 hung twice (once in 240 RPM, once in the hit probe).
- With the default 2048 prefill budget: 16K TTFT 15 s, 32K 41 s.
