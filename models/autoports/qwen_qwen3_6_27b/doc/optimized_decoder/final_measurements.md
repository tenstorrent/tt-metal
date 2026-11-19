# Final194 measurement tables

Milliseconds, warmed prefill and median traced warmed decode. All real checkpoint weights; one physical Blackhole chip. Legacy rows use matching input layout for before/after comparison. Compact rows describe the preferred public decode layout. Profiler and allocation tracking are off in performance rows.

| Kind | Batch / prefix | Fused prefill | Final prefill | Fused decode | Final decode | Decode speedup | Fused prefill / decode PCC | Final prefill / decode PCC |
|---|---|---:|---:|---:|---:|---:|---|---|
| GDN | 1 / 512 | 5.577254 | 6.213297 | 2.132502 | 1.166630 | 1.828× | 0.999970580 / 0.999997493 | 0.999945754 / 0.999991784 |
| GDN | 32 / 33 | 26.555405 | 24.413962 | 7.963243 | 1.681194 | 4.737× | 0.999963837 / 0.999972822 | 0.999948102 / 0.999937043 |
| Full attention | 1 / 512 | 5.751186 | 3.857945 | 2.041142 | 1.031517 | 1.979× | 0.999979522 / 0.999985391 | 0.999618641 / 0.999760562 |
| Full attention | 32 / 33 | 23.030737 | 13.800262 | 4.922335 | 1.185567 | 4.152× | 0.999970275 / 0.999958479 | 0.999423349 / 0.999313102 |

Baselines: baseline_0.json (successful log baseline_retry_0.log), precision_baseline_3.json, baseline_0_b32.json, baseline_3_b32.json. Final rows: final194_performance_perf_l{0,3}_b{1,32}_s{512,33}_legacy.json. Fused GDN fails the separate 512-step accuracy gate; its timing is a historical performance baseline, not the best correct long-sequence candidate.

| Final artifact | Prefill ms | Traced decode ms | Prefill PCC | Decode PCC |
|---|---:|---:|---:|---:|
| [final194_performance_perf_l0_b1_s1025_legacy](final194_performance_perf_l0_b1_s1025_legacy.json) | 14.428693 | 1.166174 | 0.999942898 | 0.999926967 |
| [final194_performance_perf_l0_b1_s33_legacy](final194_performance_perf_l0_b1_s33_legacy.json) | 3.559374 | 1.166034 | 0.999971834 | 0.999975657 |
| [final194_performance_perf_l0_b1_s512_compact](final194_performance_perf_l0_b1_s512_compact.json) | 6.303349 | 1.166249 | 0.999945754 | 0.999991784 |
| [final194_performance_perf_l0_b1_s512_legacy](final194_performance_perf_l0_b1_s512_legacy.json) | 6.213297 | 1.166630 | 0.999945754 | 0.999991784 |
| [final194_performance_perf_l0_b1_s513_legacy](final194_performance_perf_l0_b1_s513_legacy.json) | 8.260271 | 1.166069 | 0.999946060 | 0.999992930 |
| [final194_performance_perf_l0_b32_s33_compact](final194_performance_perf_l0_b32_s33_compact.json) | 24.355480 | 1.580467 | 0.999948102 | 0.999937043 |
| [final194_performance_perf_l0_b32_s33_legacy](final194_performance_perf_l0_b32_s33_legacy.json) | 24.413962 | 1.681194 | 0.999948102 | 0.999937043 |
| [final194_performance_perf_l3_b1_s1025_legacy](final194_performance_perf_l3_b1_s1025_legacy.json) | 9.514589 | 1.038220 | 0.999680114 | 0.999847710 |
| [final194_performance_perf_l3_b1_s33_legacy](final194_performance_perf_l3_b1_s33_legacy.json) | 2.657921 | 1.009515 | 0.999431303 | 0.999403695 |
| [final194_performance_perf_l3_b1_s512_compact](final194_performance_perf_l3_b1_s512_compact.json) | 3.737495 | 1.031732 | 0.999618641 | 0.999760562 |
| [final194_performance_perf_l3_b1_s512_legacy](final194_performance_perf_l3_b1_s512_legacy.json) | 3.857945 | 1.031517 | 0.999618641 | 0.999760562 |
| [final194_performance_perf_l3_b1_s513_legacy](final194_performance_perf_l3_b1_s513_legacy.json) | 5.674889 | 1.031707 | 0.999618929 | 0.999505617 |
| [final194_performance_perf_l3_b32_s33_compact](final194_performance_perf_l3_b32_s33_compact.json) | 13.636709 | 1.079028 | 0.999423349 | 0.999313102 |
| [final194_performance_perf_l3_b32_s33_legacy](final194_performance_perf_l3_b32_s33_legacy.json) | 13.800262 | 1.185567 | 0.999423349 | 0.999313102 |
| [final194_longprefill_l0_s1025](final194_longprefill_l0_s1025.json) | 364.492383 | 1.681439 | 0.999944350 | 0.999959230 |
| [final194_longprefill_l0_s512](final194_longprefill_l0_s512.json) | 175.079354 | 1.692967 | 0.999943855 | 0.999946936 |
| [final194_longprefill_l0_s513](final194_longprefill_l0_s513.json) | 182.304933 | 1.681064 | 0.999945352 | 0.999953929 |
| [final194_longprefill_l3_s1025](final194_longprefill_l3_s1025.json) | 232.038138 | 1.414905 | 0.999679143 | 0.999735273 |
| [final194_longprefill_l3_s512](final194_longprefill_l3_s512.json) | 105.056976 | 1.295772 | 0.999609529 | 0.999694923 |
| [final194_longprefill_l3_s513](final194_longprefill_l3_s513.json) | 113.352035 | 1.303898 | 0.999609855 | 0.999725759 |

| Final stress artifact | Batch | Steps | Minimum per-user PCC | Traced median ms |
|---|---:|---:|---:|---:|
| [final194_correctness_stress_l0_b1](final194_correctness_stress_l0_b1.json) | 1 | 128 | 0.999555475854 | 1.190190 |
| [final194_correctness_stress_l0_b32](final194_correctness_stress_l0_b32.json) | 32 | 512 | 0.995542745612 | 1.606030 |
| [final194_correctness_stress_l3_b1](final194_correctness_stress_l3_b1.json) | 1 | 128 | 0.998100146002 | 1.043846 |
| [final194_correctness_stress_l3_b32](final194_correctness_stress_l3_b32.json) | 32 | 512 | 0.996605367687 | 1.182385 |

| Final profile | Prefill ops | Prefill kernel + gap ms | Decode ops | Decode kernel + gap ms | Same-run host ms | Bandwidth lower bound ms | Roofline / host |
|---|---:|---:|---:|---:|---:|---:|---:|
| [final194_profile_profile_l0_b1_s1025](perf/final194_profile_profile_l0_b1_s1025/metrics.json) | 135 | 13.484025 | 33 | 1.158774 | 1.194849 | 0.925744 | 77.478% |
| [final194_profile_profile_l0_b1_s512](perf/final194_profile_profile_l0_b1_s512/metrics.json) | 46 | 5.811718 | 33 | 1.154442 | 1.185611 | 0.925744 | 78.082% |
| [final194_profile_profile_l0_b1_s513](perf/final194_profile_profile_l0_b1_s513/metrics.json) | 89 | 7.822192 | 33 | 1.157618 | 1.188577 | 0.925744 | 77.887% |
| [final194_profile_profile_l0_b32_s33](perf/final194_profile_profile_l0_b32_s33/metrics.json) | 236 | 23.604350 | 33 | 1.566280 | 1.602468 | 1.123648 | 70.120% |
| [final194_profile_profile_l3_b1_s1025](perf/final194_profile_profile_l3_b1_s1025/metrics.json) | 114 | 8.759226 | 44 | 1.039310 | 1.066113 | 0.776845 | 72.867% |
| [final194_profile_profile_l3_b1_s512](perf/final194_profile_profile_l3_b1_s512/metrics.json) | 28 | 3.199975 | 44 | 1.031187 | 1.059681 | 0.774664 | 73.104% |
| [final194_profile_profile_l3_b1_s513](perf/final194_profile_profile_l3_b1_s513/metrics.json) | 81 | 5.133317 | 44 | 1.031460 | 1.060613 | 0.774669 | 73.040% |
| [final194_profile_profile_l3_b32_s33](perf/final194_profile_profile_l3_b32_s33/metrics.json) | 442 | 13.305673 | 51 | 1.073185 | 1.101611 | 0.777240 | 70.555% |

Each profile directory contains prefill_perf_report.txt/.csv, decode_perf_report.txt/.csv, ops.csv.gz, metrics.json, and accounting.json. Prefill profiles at S512/513/1025 expose the full chunk and awkward-tail costs; numbers include profile drain and are not interchangeable with profiler-off timings. Bandwidth is512 decimal GB/s on the one assigned chip. The accounting includes physical BFP8 exponent bytes and necessary state/KV reads/writes, excludes unused prefill copies and transient intermediates. FLOPs percentages from tt-perf-report can exceed100% because its row uses reader-core count; they are not whole-chip utilization.
