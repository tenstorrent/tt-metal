# Serving benchmark

| Row | Slots | Requests in flight | Input/output | Sampling | Prefill tok/s | Decode tok/s/user | TTFT s | Step ms |
|---|---:|---:|---|---|---:|---:|---:|---:|
| b32-c32-i4096-gpqa | 32 | 32 | 4096/128 | gpqa | 624.34 | 4.31 | 6.560 | 232.235 |
| b32-c32-i22420-gpqa | 32 | 32 | 22420/128 | gpqa | 218.06 | 1.36 | 102.813 | 735.620 |
| b32-c32-i28672-gpqa | 32 | 32 | 28672/128 | gpqa | 206.39 | 1.15 | 138.923 | 866.399 |
| b32-c1-i4096-gpqa | 32 | 1 | 4096/128 | gpqa | 4565.74 | 26.36 | 0.897 | 37.930 |
| b32-c1-i22420-gpqa | 32 | 1 | 22420/128 | gpqa | 3899.41 | 19.24 | 5.750 | 51.985 |
| b32-c1-i28672-gpqa | 32 | 1 | 28672/128 | gpqa | 3687.20 | 17.66 | 7.776 | 56.637 |
| b32-c1-i131072-gpqa | 32 | 1 | 131072/128 | gpqa | 1913.57 | 7.44 | 68.496 | 134.403 |
| b32-c1-i262016-gpqa | 32 | 1 | 262016/128 | gpqa | 1182.83 | 4.28 | 221.517 | 233.487 |
| b32-c32-i4096-swe | 32 | 32 | 4096/128 | swe | 269.36 | 5.85 | 15.206 | 170.891 |
| b32-c32-i22420-swe | 32 | 32 | 22420/128 | swe | 217.91 | 1.36 | 102.889 | 735.456 |
| b32-c32-i28672-swe | 32 | 32 | 28672/128 | swe | 206.54 | 1.15 | 138.820 | 866.916 |
| b32-c1-i4096-swe | 32 | 1 | 4096/128 | swe | 4492.76 | 26.35 | 0.912 | 37.947 |
| b32-c1-i22420-swe | 32 | 1 | 22420/128 | swe | 3887.55 | 19.24 | 5.767 | 51.987 |
| b32-c1-i28672-swe | 32 | 1 | 28672/128 | swe | 3687.18 | 17.68 | 7.776 | 56.569 |
| b32-c1-i131072-swe | 32 | 1 | 131072/128 | swe | 1910.53 | 7.45 | 68.605 | 134.314 |
| b8-c8-i4096-gpqa | 8 | 8 | 4096/128 | gpqa | 1693.64 | 16.27 | 2.418 | 61.467 |
| b8-c8-i22420-gpqa | 8 | 8 | 22420/128 | gpqa | 907.18 | 5.51 | 24.714 | 181.523 |
| b8-c8-i28672-gpqa | 8 | 8 | 28672/128 | gpqa | 859.51 | 4.23 | 33.358 | 236.339 |
| b8-c1-i4096-gpqa | 8 | 1 | 4096/128 | gpqa | 5187.55 | 35.14 | 0.790 | 28.460 |
| b8-c1-i22420-gpqa | 8 | 1 | 22420/128 | gpqa | 4227.07 | 31.37 | 5.304 | 31.881 |
| b8-c1-i28672-gpqa | 8 | 1 | 28672/128 | gpqa | 3981.02 | 30.28 | 7.202 | 33.030 |
| b8-c1-i131072-gpqa | 8 | 1 | 131072/128 | gpqa | 1983.46 | 19.21 | 66.082 | 52.063 |
| b8-c8-i4096-swe | 8 | 8 | 4096/128 | swe | 1127.13 | 19.03 | 3.634 | 52.547 |
| b8-c8-i22420-swe | 8 | 8 | 22420/128 | swe | 899.14 | 5.50 | 24.935 | 181.932 |
| b8-c8-i28672-swe | 8 | 8 | 28672/128 | swe | 860.22 | 4.21 | 33.331 | 237.410 |
| b8-c1-i4096-swe | 8 | 1 | 4096/128 | swe | 5180.48 | 35.12 | 0.791 | 28.473 |
| b8-c1-i22420-swe | 8 | 1 | 22420/128 | swe | 4247.89 | 31.33 | 5.278 | 31.919 |
| b8-c1-i28672-swe | 8 | 1 | 28672/128 | swe | 3980.92 | 30.26 | 7.202 | 33.045 |
| b8-c1-i131072-swe | 8 | 1 | 131072/128 | swe | 1979.97 | 19.26 | 66.199 | 51.929 |
| b1-c1-i4096-gpqa | 1 | 1 | 4096/128 | gpqa | 5428.22 | 35.31 | 0.755 | 28.322 |
| b1-c1-i22420-gpqa | 1 | 1 | 22420/128 | gpqa | 4335.14 | 33.38 | 5.172 | 29.957 |
| b1-c1-i28672-gpqa | 1 | 1 | 28672/128 | gpqa | 4081.77 | 32.78 | 7.024 | 30.509 |
| b1-c1-i131072-gpqa | 1 | 1 | 131072/128 | gpqa | 2004.36 | 25.23 | 65.394 | 39.637 |
| b1-c1-i262016-gpqa | 1 | 1 | 262016/128 | gpqa | 1217.94 | 19.42 | 215.131 | 51.502 |
| b1-c1-i4096-swe | 1 | 1 | 4096/128 | swe | 5031.34 | 35.31 | 0.814 | 28.323 |
| b1-c1-i22420-swe | 1 | 1 | 22420/128 | swe | 4355.37 | 33.37 | 5.148 | 29.965 |
| b1-c1-i28672-swe | 1 | 1 | 28672/128 | swe | 4081.18 | 32.75 | 7.025 | 30.533 |
| b1-c1-i131072-swe | 1 | 1 | 131072/128 | swe | 2005.88 | 25.24 | 65.344 | 39.618 |
| b1-c1-i262016-swe | 1 | 1 | 262016/128 | swe | 1217.86 | 19.51 | 215.144 | 51.261 |
| release-b1-i128 | 1 | 1 | 128/128 | gpqa | 1188.60 | 38.91 | 0.108 | 25.697 |
| release-b32-i128 | 32 | 32 | 128/128 | gpqa | 147.03 | 29.32 | 0.871 | 34.110 |
| release-b1-i2048 | 1 | 1 | 2048/128 | gpqa | 5565.16 | 35.53 | 0.368 | 28.146 |
| release-b1-i8192 | 1 | 1 | 8192/128 | gpqa | 5146.50 | 34.77 | 1.592 | 28.762 |
| release-b1-i32768 | 1 | 1 | 32768/128 | gpqa | 3930.01 | 32.36 | 8.338 | 30.906 |

Release matrix: main 7a51c2f20f51e7508d8ea365d09092d00bbd5fcb release_model_spec.json google/gemma-4-31B-it P300X2 tt_transformers (matrix only; autoport runtime unchanged). Only its five text rows apply to this text-only run. No implementation or serving config was borrowed.

## Predictions and full-phase accounting

| Row | Predicted decode range | Measured decode | Predicted prefill range | Measured prefill | Prefill host-wall FLOP % | Decode host-wall DRAM % |
|---|---:|---:|---:|---:|---:|---:|
| b32-c32-i4096-gpqa | 21.27–35.44 | 4.31 | 95.98–159.97 | 624.34 | 11.401 | 25.222 |
| b32-c32-i22420-gpqa | 19.34–32.23 | 1.36 | 86.34–143.90 | 218.06 | 10.073 | 20.297 |
| b32-c32-i28672-gpqa | 18.40–30.67 | 1.15 | 81.70–136.17 | 206.39 | 9.803 | 19.464 |
| b32-c1-i4096-gpqa | 21.27–35.44 | 26.36 | 3071.37–5118.95 | 4565.74 | 11.242 | 24.337 |
| b32-c1-i22420-gpqa | 19.34–32.23 | 19.24 | 2762.85–4604.76 | 3899.41 | 10.036 | 18.530 |
| b32-c1-i28672-gpqa | 18.40–30.67 | 17.66 | 2614.54–4357.57 | 3687.20 | 9.760 | 17.236 |
| b32-c1-i131072-gpqa | 12.59–20.98 | 7.44 | 1731.53–2885.88 | 1913.57 | 7.454 | 8.888 |
| b32-c1-i262016-gpqa | 8.86–14.76 | 4.28 | 1194.27–1990.45 | 1182.83 | 6.519 | 6.291 |
| b32-c32-i4096-swe | 21.27–35.44 | 5.85 | 95.98–159.97 | 269.36 | 11.358 | 23.186 |
| b32-c32-i22420-swe | 19.34–32.23 | 1.36 | 86.34–143.90 | 217.91 | 10.073 | 20.273 |
| b32-c32-i28672-swe | 18.40–30.67 | 1.15 | 81.70–136.17 | 206.54 | 9.811 | 19.374 |
| b32-c1-i4096-swe | 21.27–35.44 | 26.35 | 3071.37–5118.95 | 4492.76 | 11.070 | 24.319 |
| b32-c1-i22420-swe | 19.34–32.23 | 19.24 | 2762.85–4604.76 | 3887.55 | 10.008 | 18.524 |
| b32-c1-i28672-swe | 18.40–30.67 | 17.68 | 2614.54–4357.57 | 3687.18 | 9.764 | 17.249 |
| b32-c1-i131072-swe | 12.59–20.98 | 7.45 | 1731.53–2885.88 | 1910.53 | 7.443 | 8.888 |
| b8-c8-i4096-gpqa | 24.20–40.33 | 16.27 | 383.92–639.87 | 1693.64 | 12.519 | 35.359 |
| b8-c8-i22420-gpqa | 22.01–36.68 | 5.51 | 345.36–575.59 | 907.18 | 10.897 | 33.695 |
| b8-c8-i28672-gpqa | 20.94–34.90 | 4.23 | 326.82–544.70 | 859.51 | 10.556 | 33.559 |
| b8-c1-i4096-gpqa | 24.20–40.33 | 35.14 | 3071.37–5118.95 | 5187.55 | 12.320 | 32.615 |
| b8-c1-i22420-gpqa | 22.01–36.68 | 31.37 | 2762.85–4604.76 | 4227.07 | 10.828 | 30.253 |
| b8-c1-i28672-gpqa | 20.94–34.90 | 30.28 | 2614.54–4357.57 | 3981.02 | 10.498 | 29.573 |
| b8-c1-i131072-gpqa | 14.32–23.87 | 19.21 | 1731.53–2885.88 | 1983.46 | 7.724 | 22.739 |
| b8-c8-i4096-swe | 24.20–40.33 | 19.03 | 383.92–639.87 | 1127.13 | 12.458 | 34.875 |
| b8-c8-i22420-swe | 22.01–36.68 | 5.50 | 345.36–575.59 | 899.14 | 10.832 | 33.577 |
| b8-c8-i28672-swe | 20.94–34.90 | 4.21 | 326.82–544.70 | 860.22 | 10.561 | 32.937 |
| b8-c1-i4096-swe | 24.20–40.33 | 35.12 | 3071.37–5118.95 | 5180.48 | 12.297 | 32.604 |
| b8-c1-i22420-swe | 22.01–36.68 | 31.33 | 2762.85–4604.76 | 4247.89 | 10.874 | 30.246 |
| b8-c1-i28672-swe | 20.94–34.90 | 30.26 | 2614.54–4357.57 | 3980.92 | 10.496 | 29.568 |
| b8-c1-i131072-swe | 14.32–23.87 | 19.26 | 1731.53–2885.88 | 1979.97 | 7.713 | 22.739 |
| b1-c1-i4096-gpqa | 26.40–44.00 | 35.31 | 3071.37–5118.95 | 5428.22 | 12.760 | 32.849 |
| b1-c1-i22420-gpqa | 24.01–40.02 | 33.38 | 2762.85–4604.76 | 4335.14 | 11.083 | 32.273 |
| b1-c1-i28672-gpqa | 22.84–38.07 | 32.78 | 2614.54–4357.57 | 4081.77 | 10.755 | 32.048 |
| b1-c1-i131072-gpqa | 15.62–26.04 | 25.23 | 1731.53–2885.88 | 2004.36 | 7.805 | 29.739 |
| b1-c1-i262016-gpqa | 11.00–18.33 | 19.42 | 1194.27–1990.45 | 1217.94 | 6.710 | 28.010 |
| b1-c1-i4096-swe | 26.40–44.00 | 35.31 | 3071.37–5118.95 | 5031.34 | 12.728 | 32.333 |
| b1-c1-i22420-swe | 24.01–40.02 | 33.37 | 2762.85–4604.76 | 4355.37 | 11.132 | 32.275 |
| b1-c1-i28672-swe | 22.84–38.07 | 32.75 | 2614.54–4357.57 | 4081.18 | 10.748 | 32.051 |
| b1-c1-i131072-swe | 15.62–26.04 | 25.24 | 1731.53–2885.88 | 2005.88 | 7.812 | 29.736 |
| b1-c1-i262016-swe | 11.00–18.33 | 19.51 | 1194.27–1990.45 | 1217.86 | 6.710 | 28.008 |
| release-b1-i128 | 26.98–44.97 | 38.91 | 4196.64–4196.64 | 1188.60 | 2.879 | 35.210 |
| release-b32-i128 | 21.73–36.22 | 29.32 | 131.15–131.15 | 147.03 | 8.790 | 32.087 |
| release-b1-i2048 | 26.70–44.49 | 35.53 | 4146.92–4146.92 | 5565.16 | 13.047 | 32.919 |
| release-b1-i8192 | 25.82–43.04 | 34.77 | 3995.43–3995.43 | 5146.50 | 12.300 | 32.624 |
| release-b1-i32768 | 22.84–38.07 | 32.36 | 3486.06–3486.06 | 3930.01 | 10.545 | 31.936 |

Headline sampling-mode step counts (all grid rows retain their own counts and per-step evidence):

| Batch | Host-compatible prefill | Device prefill | Device decode |
|---|---:|---:|---:|
| 32 | 96 | 96 | 416 |
| 8 | 24 | 24 | 392 |
| 1 | 8 | 8 | 1016 |

Predictions were recorded in ../plan.json before the first server launch. Raw per-request ITL, E2E, percentiles, completion counts, throughput and duration remain beside each row. Each row in ../serving_rates.json links its raw timings, accounting and any prediction miss.
The original prefill estimate describes aggregate capacity. For comparison with input/mean per-request TTFT, its rate and bounds are divided by concurrency, consistent with the original serialized-prefill prediction basis. Original aggregate estimates remain in predicted.prefill_aggregate_original; this unit conversion uses no measured values. Decode predictions already have per-user units. Row causes quantify prefill interference and measured decode intervals without claiming device-level attribution.
For stage12 explain_hours.py, pass --rates ../serving_rates-gpqa.json for seed42 traffic or ../serving_rates-swe.json for unseeded SWE traffic. These are exact policy-filtered views of serving_rates.json; the unmodified interpolator does not select a sampling policy itself. Do not interpolate across both policies together.
Infinite-rate burst with bounded concurrency. Native headline uses max(8,3*batch) requests; other grid rows use max(2,concurrency). Warmup is one concurrency wave. GPQA/SWE headline comparisons therefore differ in population and are not a seed-only experiment.

Phase time is the nonoverlapping wall interval between consecutive adapter-observed actual completions. Plugin host sampling, scheduling, output construction and idle gaps after an adapter return fall in the next interval. The first interval starts at the first measured request; the final client tail belongs to the last phase. This is operator decision 4 attribution, not separately observed plugin timing. FLOPs and DRAM estimates use executed shapes, precision and TP4; both numerators and denominators cover the same measured requests. No profiler or added device synchronization. Device-time utilization is unavailable.

## Named gaps

- b32-c32-i131072-gpqa: Live unchanged B32 server reports 1973488 cache tokens; 32 * (131072+128) = 4198400 exceeds it. Concurrency1 remains scheduled.
- b32-c32-i262016-gpqa: Live unchanged B32 server reports 1973488 cache tokens; 32 * (262016+128) = 8388608 exceeds it. Concurrency1 remains scheduled.
- b32-c32-i131072-swe: Live unchanged B32 server reports 1973488 cache tokens; 32 * (131072+128) = 4198400 exceeds it. Concurrency1 remains scheduled.
- b32-c32-i262016-swe: Live unchanged B32 server reports 1973488 cache tokens; 32 * (262016+128) = 8388608 exceeds it. Concurrency1 remains scheduled.
- b32-c1-i262016-swe: Insufficient remaining stage budget for warmup, complete measurement and required closure; no substituted result.
- b8-c8-i131072-gpqa: Unchanged live B8 KV pool has 548143 token capacity; 1049600 tokens required for the simultaneous prompt/output working set. Full context remains262144; concurrency1 rows are retained.
- b8-c8-i262016-gpqa: Unchanged live B8 KV pool has 548143 token capacity; 2097152 tokens required for the simultaneous prompt/output working set. Full context remains262144; concurrency1 rows are retained.
- b8-c1-i262016-gpqa: Insufficient remaining stage budget for warmup, complete measurement and required closure; no substituted result.
- b8-c8-i131072-swe: Unchanged live B8 KV pool has 548143 token capacity; 1049600 tokens required for the simultaneous prompt/output working set. Full context remains262144; concurrency1 rows are retained.
- b8-c8-i262016-swe: Unchanged live B8 KV pool has 548143 token capacity; 2097152 tokens required for the simultaneous prompt/output working set. Full context remains262144; concurrency1 rows are retained.
- b8-c1-i262016-swe: Insufficient remaining stage budget for warmup, complete measurement and required closure; no substituted result.

Model: full60-layer generated autoport, pinned model/tokenizer842da3794eaa0b77d5f08bae87a17459d91ff475, head_bfp4, P300x2/TP4, full context262144, prefix caching enabled. Per-profile identities retain actual launch argv, source hashes, module and revision evidence. Each invocation uses a unique cache salt and distinct dataset seed, matching main benchmark cache isolation; production prefix caching stays enabled. Server defaults remain temperature1/top_p0.95/top_k32. Benchmarks send eval top_k20, with seed42 for GPQA and no seed for SWE. All performance requests ignore EOS.

## Anomaly and logging checks

See ../anomaly_check.json, ../semantic_concerns.json, ../qualitative_verdict.json and ../../agentic_evals/logging_check.json. Detector flags measure mechanical degeneration, not answer correctness. Independent review identified acoustic-frequency, routing-feasibility and calibration-example inconsistencies in three generic replies; these semantic concerns are recorded for stage12 without attributing them to TT numerics. Stage11 makes no eval score or precision change.

Four sampled replies naturally stopped after 1413, 1654, 1624 and 3929 generated tokens at a 32768-token cap, using served defaults temperature1/top_p0.95/top_k32. These checks cover 8620 generated tokens in total; they do not demonstrate sustained generation to the eval output caps or a ten-minute soak. Stage12 must assess longer-output behavior. All four complete response bodies match the server logs.

## Timing and provenance

Completed grid bundles carried from the budget-handoff attempt remain byte-exact. setup/native_fallback.json maps their discovery copies to archived originals and hashes; embedded old absolute server paths refer to that archived attempt, not the new server. The canonical native summary is a genuine fresh all-batch run, not merged results.
The installed vLLM distribution is 0.31.0+empty and exposes no source Git SHA. identity.json leaves that commit null and includes the installed Python-source fingerprint and distribution metadata hashes; tt-metal, plugin and main harness commits are recorded. No unverified vLLM commit is inferred.
Elapsed through report: 9921.7s / 10800s. All server launches, warmup and measurements are inside this clock. Setup before the clock repaired host paths, verified main's source and ran16 host accounting tests. No earlier v3/v4 results were reused.

Closure: independent review clean-pass; all three gates and repository hooks passed; no device owners remain. Evidence commit `45d940bf7233bb52824243f268a23cb70f57f75c` on `v5-extended-run` (local only). Elapsed through review and evidence commit: 10145.0s /10800s.
