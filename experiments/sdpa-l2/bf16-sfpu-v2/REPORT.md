# FAST BF16 SFPU scheduling optimization

## Result

The retained change reduces the Q256/K512 single-core resident-input loop's time by **5.06%**, increases useful throughput by **5.33%**, and preserves the tested outputs bit-for-bit. FAST's compute-bound overhead versus main falls from **35.87% to 29.00%**. This removes **19.16% of the excess time**, not 19% of total runtime.

No Q/K chunk sizes, Q/K/V input buffering, circular-buffer capacities, datamovement kernels, fidelity, exp approximation, preprocessing, or compensation arithmetic changed. FAST still compensates both numerator and denominator. ACCURATE's algorithm is unchanged.

## Single-core, no-recurring-DM measurement

Blackhole P100A on `yyzo-bh-08`, reservation `219338`, logical core (0,0), physical (1,2), 1350 MHz. Same resident-input harness as `../single-core-resident-v1`: BF16 Q/K/V, D128, Q256/K512, original double-buffered BF16 inputs. Initial input loading is outside the steady loop; no recurring DRAM/NoC input transfers or L1 input copying. Sixteen Q iterations and 512 K chunks, 20 warmup trace replays and 10 measured blocking trace replays.

| Implementation | Median ms | Useful TFLOP/s/core | HiFi2 peak utilization | Time overhead vs main |
|---|---:|---:|---:|---:|
| Frozen main BF16 | 275.679791 | 1.994182 | 72.13% | — |
| Previous FAST | 374.567614 | 1.467708 | 53.09% | 35.87% |
| Optimized FAST | 355.616894 | 1.545922 | 55.91% | 29.00% |

Useful FLOPs are `4 * 256 * 512 * 128 * 16 * 512 = 549,755,813,888`. Utilization divides useful throughput by the same 2.7648 TFLOP/s/core HiFi2 issue-rate peak used in the previous report; it is not a hardware activity counter or measured chip throughput. This benchmark repeats resident K/V, so its L2 values are not general-input accuracy qualification.

Reversing measurement order gave **355.626810 ms**, within 0.003% of the first optimized run. Frozen controls are newly measured on the same allocation, not imported from a different machine. Raw controls are `../single-core-resident-v1/sfpu-v2-{main,fast}-control.json`; new results are `final-steady.json` and `final-reverse.json`.

### Device profile

An independent eight-Q-iteration run measured **240,201,964 cycles / 177.927381 ms**, or **1.544888 TFLOP/s/core**, within 0.07% of the uninstrumented throughput. The window is below the 32-bit counter wrap limit.

| Counter fraction | Previous FAST | Optimized FAST |
|---|---:|---:|
| FPU active | 57.753% | 60.814% |
| SFPU active | 40.855% | 37.956% |
| FPU and SFPU active | 18.307% | 19.338% |
| SFPU active without FPU | 22.548% | 18.618% |
| Neither active | 19.699% | 20.568% |

The previous profile is the frozen v1 run on this same allocation. Counter percentages use their whole-kernel reference windows; effective FLOP utilization above uses useful attention FLOPs. These are different measures. Raw new profile: `profile-final/reports/2026_09_14_17_14_52/profile_log_device.csv`, parsed in `profile-summary.jsonl`.

## Changes and ablations

The two-state compensator's replay body decreases from **21 to 15 issued instructions per vector**. Load macros overlap low-component loads with high-plus-low addition, and overlap BF16 rounding with stores. The independent chains provide the necessary scheduling gaps. Full sums remain in separate SFPU registers until residual subtraction, preserving the exact update order and the existing nearest-rounding instruction, including its tie behavior.

Macro setup, address-modifier setup, and the paired replay program are initialized once per numerator correction group. The one-state denominator retains the original 11-instruction schedule. Every final paired call drains pending macro operations before PACK consumes DST. The next correction group restores the paired program; exp initialization restores exp's macro configuration before its next use.

| Incremental experiment | Resident median ms |
|---|---:|
| Original paired replay | 374.567614 |
| Macro rounding/stores only | 363.226910 |
| Also fuse low loads with addition | 357.409874 |
| Hoist macro configuration | 356.158981 |
| Also cache paired replay and hoist address setup | 355.617934 |
| Cleaned retained implementation | 355.616894 |

The initial attempt to apply the same macro schedule to a single state was invalid: it lacked the second chain's dependency/store separation and failed the smoke test (44.585% L2). It was discarded, not benchmarked as a valid candidate. One intermediate duplicate-definition compile failure was also corrected before measurements. The retained code leaves one-state arithmetic and scheduling unchanged.

## Numerical validation

**30/30 direct state-update cases matched every output bit**, comparing both stored high and low BF16 components against the frozen implementation: one/two states, three seeds, normal, nonidentity rescaling, cancellation, wide exponents, and rounding-boundary inputs. See `state-final.jsonl` and `state_probe.py`.

**32/32 SDPA cases matched every tested output bit**, and therefore have identical L2 and PCC: N32768/N262144, H2, D128, seeds 1236/1237, normal, scaled Q/K, outliers, common Q/K/V, constant V, and uniform attention. Q128/K512, no Q preprocessing, both compensation terms enabled. These are noncausal sampled-query executions: full Q/K/V are generated, 128 spread Q rows are executed, and every executed output is compared by hash. This is an equivalence test, not a claim that FAST meets an accuracy threshold on the stress cases. See `comparison.json`, `compare.py`, and `qualify.sh`.

The ACCURATE resident holdout also retained its exact output hash and **0.1788797% L2**, **0.9999984003 PCC** (`accurate-regression.json`, compared with `../single-core-resident-v1/holdout-accurate-v1.json`).

## Full-operator measurement

Full noncausal N262144, H10, D128, normal BF16 Q/K/V, seed 1236, no preprocessing, unchanged production Q128/K512 geometry and input buffering, all 110 compute cores. Forty warmup and ten measured blocking trace replays, excluding compilation, transfers, and reference computation.

| Implementation | Median ms | Useful TFLOP/s | L2 % | PCC |
|---|---:|---:|---:|---:|
| Previous FAST | 2781.490737 | 126.495 | 3.159530 | 0.999711225 |
| Optimized FAST | 2735.552738 | 128.619 | 3.159530 | 0.999711225 |

Time decreases **1.65%** (throughput increases 1.68%). Measured ranges do not overlap: old **2772.476–2784.894 ms**, new **2724.803–2740.961 ms**. This is a single sustained run per version, not a multi-device qualification. Both were freshly measured on this allocation; do not compare their absolute latencies directly with earlier runs on other machines.

The 512 spread reference rows have exactly the same output hash, L2, and PCC. The new run additionally checks finiteness and eager-versus-trace equality of the entire output. The old run did not record an entire-output hash, so whole-output old-versus-new equality is not claimed for this full shape. Raw records: `old-fast-full.jsonl`, `new-fast-full.jsonl`.

Useful FLOPs are `4 * 10 * 262144^2 * 128 = 351.84372088832e12`. As expected, some of the no-DM improvement is hidden in the full operator. Neither version changes the datamovement configuration to obtain these results.

## Engineering conclusion

This is a worthwhile, precision-preserving scheduling improvement, but it does not make FAST nearly as cheap as main when compute-bound. Most of the original overhead remains. Further large gains likely require reducing the exposed cost of moving/packing compensated state or improving overlap around state updates, not just removing a few more SFPU issue instructions. That is an inference from the measured residual gap and counter profile, not a measured speedup for an unimplemented change.

## Source and verification

Only two production files change relative to the pre-task FAST implementation: `ckernel_sfpu_sdpa.h` and `compute_streaming.hpp`. The exact incremental change is `incremental.patch`; pre-task copies are `before-sfpu.h` and `before-streaming.hpp`. Existing unrelated/earlier project edits are preserved. Factory guards remain unchanged, including their Q128 restriction; the existing resident harness explicitly exercises Q256 without claiming that production automatically selects this path at Q256.

Final production SHA256:

```
826eb19a33042953b7d1d617dbd5fcb0452d0d552604d4033b5f8e19ab299578  ckernel_sfpu_sdpa.h
b891cf15e8b48a1ccbed4d8397b6573cc1e1e51ee3956893da8afc0f24f40ea7  compute_streaming.hpp
```

The remote checkout initially held older project edits; these were archived in `remote-before.tar` before installing the current local baseline for full-operator comparisons. The final remote production sources match the local retained implementation.

Build passed using `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install` in the configured IRD container (`build.log`, `final-build.log`), and all retained device kernels compiled through JIT and ran on hardware. Production `clang-format --dry-run --Werror`, Python Black checks targeting Python 3.10, shell syntax checks, and local `git diff --check` passed. Reproduction scripts are `final_measure.sh`, `qualify.sh`, `state_probe.py`; use the frozen v1 driver for main and pre-optimization FAST controls.
