# FAST: pair denominator compensation rows

## Change

Two denominator rows now share a DST acquire/commit/wait/release cycle. Each row retains its own high component, low component, new chunk, and correction: eight BF16 tiles in the existing half-DST. The state-update helper uses a stride of two for the interleaved denominator rows. The number of copied/packed state tiles is unchanged, but copy/broadcast initialization and DST handoffs are shared.

The two independent SFPU chains use a 16-instruction replay program in entries 15–30, beside the numerator's existing 15-instruction program in entries 0–14. Both reuse the same load/add and round/store macros. The second denominator chain reads its separate correction from L7; it does not share the first row's scale. Two old one-row updates required 22 replay-body instructions per vector.

No Q/K chunks, input double buffering, CB capacities or data formats, datamovement kernels, exp coefficients, fidelity, preprocessing, or compensation arithmetic changed. Odd final rows retain the original one-row update. ACCURATE's algorithm is unchanged. Factory guards are unchanged.

## Initial no-DM screen

On reservation 219338, `yyzo-bh-08`, Blackhole P100A, logical core (0,0), physical (1,2). Q256/K512/D128, BF16 inputs, sixteen Q iterations, 512 K chunks, 20 warmup and ten measured blocking trace replays. Inputs are preloaded in the same double-buffered resident ring; there are no recurring DRAM/NoC reads or L1 input copies.

| Version | Median ms | Useful TFLOP/s/core |
|---|---:|---:|
| Previous FAST with numerator macros | 355.621365 | 1.545902 |
| Paired denominator | 345.443914 | 1.591447 |

This is **2.862% less time / 2.946% more throughput**. The output hash, L2, and PCC match exactly. These repeated resident inputs are for scheduling/performance evaluation, not a general-input accuracy suite.

Raw records: `paired-steady-v1.json` and `../bf16-sfpu-v2/denom-v3-control.json`. Useful FLOPs are `4 * 256 * 512 * 128 * 16 * 512 = 549,755,813,888`.

## Final no-DM measurements

| Version | Median ms | Useful TFLOP/s/core | HiFi2 peak utilization | Overhead vs main |
|---|---:|---:|---:|---:|
| Frozen main BF16 | 275.672683 | 1.994234 | 72.13% | — |
| Previous FAST with numerator macros | 355.623442 | 1.545893 | 55.91% | 29.00% |
| Retained paired-denominator FAST | 345.443981 | 1.591447 | 57.56% | 25.31% |

Utilization divides useful throughput by the same 2.7648 TFLOP/s/core HiFi2 peak at 1350 MHz as the previous report. It is not a hardware activity-counter percentage or measured chip throughput.

The cleaned implementation reproduces the initial gain. A reverse-order repeat gives **345.451476 ms**, within 0.003%. Controls are freshly measured on the same allocation. Relative to FAST before either SFPU optimization (374.567614 ms in the preceding investigation), the two changes together reduce resident time by **7.78%**. Significant compute-bound overhead remains; this is an incremental improvement, not parity with main.

Raw new records: `final-steady.json`, `final-reverse.json`; controls: `../bf16-sfpu-v2/denom-v3-control-repeat.json`, `../single-core-resident-v1/denom-v3-main.json`.

### Device profile

An independent eight-Q-iteration profile measures **233,128,587 cycles / 172.687842 ms**, or **1.591762 TFLOP/s/core**, within 0.02% of uninstrumented throughput. The profile window is below the 32-bit counter wrap limit.

| Hardware activity fraction | Previous FAST | Paired denominator |
|---|---:|---:|
| FPU active | 60.814% | 62.660% |
| SFPU active | 37.956% | 37.782% |
| Both active | 19.338% | 20.219% |
| SFPU only | 18.618% | 17.563% |
| Neither active | 20.568% | 19.778% |

The previous profile is from v2 on the same allocation. Raw new counters are in `profile-final/reports/2026_09_14_18_20_16/profile_log_device.csv`, parsed in `profile-summary.jsonl`. These counter fractions use the whole-kernel reference window; effective FLOP utilization above uses useful attention FLOPs.

## Numerical validation

**45/45 direct state-update cases match every output bit**, including both BF16 high and low components. **36/36 SDPA comparisons also match**: all executed outputs in the 32 noncausal sampled-query cases, and complete outputs in four causal full-sequence cases. See `state-final.jsonl` and `comparison.json`.

The noncausal matrix covers N32768/N262144, H2, D128, seeds 1236/1237, and normal, scaled QK, outliers, common Q/K/V, constant V, and uniform attention. It executes 128 spread query rows from full generated inputs, with Q128/K512 and no preprocessing. The causal cases cover N32768/N65536, H2, D128, normal/scaled QK, seed 1236, with the same chunks. This validates equivalence, not accuracy-threshold acceptance for FAST's stress cases.

The ACCURATE resident regression retains its exact output hash, **0.1788797% L2**, and **0.9999984003 PCC** (`accurate-regression.json`, matching the frozen seed-1237/K64 holdout).

## Full-operator performance

Full noncausal N262144, H10, D128, normal BF16 Q/K/V, seed 1236, original Q128/K512 production configuration, no preprocessing, all 110 compute cores. Forty warmup and ten measured blocking trace replays, excluding compilation, reference calculation, and transfers.

| Version | Median ms | Useful TFLOP/s | L2 % | PCC |
|---|---:|---:|---:|---:|
| Previous FAST with numerator macros | 2734.737199 | 128.657 | 3.159530 | 0.999711225 |
| Paired denominator | 2725.778444 | 129.080 | 3.159530 | 0.999711225 |

The median decreases **0.33%**, but ranges overlap: old **2724.390–2744.150 ms**, new **2721.456–2738.590 ms**. This is one sustained run per version, and the small median difference is not sufficient to claim a robust full-chip speedup. The clear result is the reproducible 2.86% no-DM improvement; most of that benefit is hidden at this full-operator configuration.

The **entire output hash** matches between old and new, and both pass complete-output finiteness and eager-versus-trace equality checks. L2 and PCC use 512 spread reference rows. This is an additional full-shape equivalence check beyond the 36-case suite above. Raw records: `old-full.jsonl`, `new-full.jsonl`. Useful FLOPs are `4 * 10 * 262144^2 * 128 = 351.84372088832e12`.

## Reproduction and source

Only two production files change relative to the previous task: the Blackhole `ckernel_sfpu_sdpa.h` and SDPA `compute_streaming.hpp`. See `incremental.patch`. Pre-task copies are `before-sfpu.h` and `before-streaming.hpp`; other existing project edits are preserved.

`state_probe.py` tests one row, two shared-correction states, and two independently corrected rows. For the independent-row reference it invokes the old one-state kernel separately on each row, then concatenates the results. Each mode covers three seeds and normal, rescale, cancellation, wide-exponent, and rounding-boundary inputs: 45 cases total.

`qualify.sh new` repeats the previous task's 32 noncausal sampled-query SDPA cases. `compare.py` explicitly compares with the **final v2** records (`../bf16-sfpu-v2/new-*.jsonl`), not the older pre-macro controls. `extra_qualify.sh old/new` additionally checks complete outputs for four causal cases: N32768/N65536, H2, normal/scaled QK, D128, Q128/K512. `final_measure.sh` runs fresh no-DM controls, reverse-order repetition, an ACCURATE regression case, the FPU/SFPU counters, and full N262144/H10 noncausal attention.

Production Q128/K512 is unchanged for full-operator measurements. Q256 is exercised explicitly by the existing resident harness; the production Q128 selection guard is not broadened by this patch.

Build passed with `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install` (`build.log`); retained kernels also compile through device JIT. Production clang-format, Python Black checks targeting Python 3.10, shell syntax checks, and local `git diff --check` passed.

Final production SHA256:

```
54c51c56aa457134aed0181202c6fd3cad66df0a507659f364857c77dca9bb42  ckernel_sfpu_sdpa.h
b471e527b61f55f2c9f30573ee4cdd82f5cbad7d0b02e1c654814835d3d0916b  compute_streaming.hpp
```
