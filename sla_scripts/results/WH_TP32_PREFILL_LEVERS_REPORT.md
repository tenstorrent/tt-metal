# Llama-3.3-70B, TP32 on one WH Galaxy, prefill levers: results vs customer SLA

Code:
- tt-metal `gtobar/prefill-levers` `3dd62e0b738` (no local changes)
- vllm-tt-plugin `gtobar/prefill-arrival-wait` `dc08531` (fork gtobarTT/vllm-tt-plugin)

Run: `vllm/run_prefill_levers_suite.sh C`. Env, flags and knobs: see the script.
No precision change: bfp4 weights and HiFi2 SDPA stay off.

## Configs

| Config | Async scheduling | Plugin knobs | Results |
|---|---|---|---|
| A | off | none | `bench_prefill_levers_async_off/` |
| B | on | flush, admission cap 1, arrival wait 2 ms | `bench_prefill_levers_async_plugin_knobs/` |
| C (best) | on | none | `bench_prefill_levers_async_on/`, `mmlu_prefill_levers_async_on/` |

- B: decode ~10 tok/s/user even for 1 user. Plugin knob values were guessed; not usable as run.
- Knobs left off:
  - `CPF_PAD512`, `CPF_REPLICA_MISS`: trace region overflow at startup (572 MB, then 663 MB).
  - `FF2_BFP4`, `ATTN_W_BFP4`, `PREFILL_SDPA_HIFI2`: change precision.

## Config C vs SLA

| SLA line | Target | Before (`WH_TP32_VLLM_REPORT.md`) | Config C |
|---|---|---|---|
| 1 user at 4K: TTFT / tok/s | 1 s / 25 | 0.69 s / 49.7 | 0.42 s / 55.2 |
| Cold 16K / 32K TTFT | 5 / 10 s | 2.14 / 4.20 s | 2.00 / 3.91 s |
| 15 RPM: TTFT mean / P99 | 1 / 5 s | 1.18 / 3.12 s | 0.59 / 1.33 s |
| 15 RPM: tok/s mean / P99 | 25 / 15 | 35.0 / 22.7 | 45.6 / 34.9 |
| 15 RPM: end-to-end mean | 20 s | 15.0 s | 11.2 s |
| 60 RPM: TTFT mean / P99 | 1 / 5 s | 1.69 / 7.02 s | 0.88 / 2.11 s |
| 60 RPM: tok/s mean / P99 | 25 / 15 | 13.5 / 10.1 | 26.1 / 19.4 |
| 60 RPM: end-to-end mean / P99 | 20 / 140 s | 39.5 / 63.3 s | 20.5 / 36.6 s |
| 60 RPM + 40% prefix: TTFT mean / P99 | 1 / 5 s | 1.00 / 3.03 s | 0.74 / 1.58 s |
| 60 RPM + 40% prefix: tok/s mean / P99 | 25 / 15 | 14.5 / 10.9 | 26.4 / 18.9 |
| 60 RPM + 40% prefix: end-to-end mean | 20 s | 34.2 s | 19.1 s |
| Prefix hit rate | 40% | 38.7% | 39% |
| 240 RPM: TTFT mean / P99 | 1 / 5 s | 3.12 / 23.7 s | 1.18 / 7.30 s |
| 240 RPM: tok/s mean / P99 | 25 / 15 | 13.3 / 9.6 | 20.7 / 16.0 |
| Users per Galaxy | 128 | 32 | 32 |
| MMLU | >= 80% | 82.3% | 82.2% on 12,631 answered |

- Correctness: needle recall 21/21 (1K-29K, cold, full and partial prefix hits); garbage probe clean.
- MMLU stopped at 90%: one request hung past the 600 s timeout (request stall bug). 73.9% if unanswered count as wrong.
- 240 RPM runs at max concurrency 32, so the served arrival rate is below 240 RPM.
- Quad (4 x 32 users): 15 RPM rows = mean load, 60 RPM rows = peak load per Galaxy. Estimate only.

## Open

- Request stall under load: not filed.
- Config A vs C: async on gives +9% decode, ~0.15 s more TTFT.
- Reported numbers from the branch author are higher (60 RPM 28.6 / 20.9 tok/s). His env and plugin values are not known.
