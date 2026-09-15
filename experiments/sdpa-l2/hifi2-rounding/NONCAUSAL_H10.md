# Non-causal full-256K remeasurement, H=10

September 9, 2026. Same Blackhole P100A, reservation 214149.

## Outcome

The earlier small accuracy/performance overhead was specific to the
bandwidth-limited causal workload. **For full non-causal attention with ten
heads, the candidate is 64.1% slower than unmodified-main FP32 and about 2x
BF16 streaming.** It still meets the sampled aggregate accuracy target at
0.4884% L2, but does not meet the low-performance-overhead objective here.

## Matched configuration and timing

B=1, Q=K=262144, H=10, D=128, Q/K chunks 128/512, non-causal, HiFi2, both
approximation flags true, seed 1236, normal inputs. All runs use the same BF16
Q preprocessing (`c=1.0027`, bitwise implementation) and compensated scale;
the reference uses original BF16 inputs. The entire square operation executes.
Accuracy samples the last 128 rows/head, against FP64 blockwise softmax.

`--full` was added to the repro to enable square non-causal attention while
retaining bounded reference sampling. Simply passing `--q-len 262144` would
also request a prohibitively large full reference and is not what was done.

| Path | Median warmed trace replay, ms | Sampled aggregate L2 % |
|---|---:|---:|
| Main BF16 destination / streaming | 2702.62 | 18.9641 |
| Main FP32 destination / standard | 3287.43 | 9.0804 |
| Improved FP32 destination / standard | 5395.74 | 0.4884 |

Seven timed blocking trace replays follow two warmups. Compilation, transfers,
reference and CPU preprocessing are excluded. Trace and ordinary outputs
matched exactly. CPU preprocessing took 106–112 ms for H=10.

BF16 streaming was also measured with the candidate sources present: 2674.38 ms,
with exactly the same accuracy. The new C++ path is gated off for BF16 destination.
Its timed samples rose from 2631.59 to 2710.80 ms; the main-source repeat rose
from 2656.02 to 2733.26 ms. FP32 candidate samples were stable at 5395.20–5396.77 ms;
main FP32 samples were 3282.74–3296.25 ms. Avoid excessive precision in overhead
claims given the BF16 drift.

Main FP32 is already 21.6% slower than BF16 streaming. The accuracy changes add
another 64.1% relative to main FP32, producing a total 99.6% increase over the
main-source BF16 streaming median. These are distinct comparisons.

For comparison, full non-causal H=4 was also tested: streaming 1341.14 ms,
improved FP32 2164.24 ms (+61.4%), with L2 18.7982% and 0.4904% respectively.
That uses five timed replays and the same D/chunks/seed/preprocessing.

## Hardware FPU/SFPU utilization

Separate single-operation captures used `--profiler-capture-perf-counters=fpu`.

| Path | Full-chip FPU active cycles % | Full-chip SFPU active cycles % | Per-core FPU median % |
|---|---:|---:|---:|
| BF16 streaming | 48.01 | 14.71 | 58.58 |
| Main FP32 | 41.15 | 13.72 | 47.26 |
| Improved FP32 | 22.08 | 40.75 | 24.19 |

Full-chip averages use 120 maximum compute cores and the whole operation's
64-bit timestamp span. The program uses 110 cores. The per-core median uses
each active core's own elapsed interval. These measure active instruction
cycles, not useful FLOPs divided by peak FLOPs.

BF16 per-core FPU utilization spans 52.32–68.64%. Thus the fastest-completing
cores approach 70%, but the full operation does not. Per-core kernel durations
span 1.806–2.370 s in this capture; inactive cores and the completion tail lower
the full-chip average. Merely normalizing by 110 instead of 120 cores would
raise 48.01% to 52.37%, not 70%.

Streaming's FPU utilization rises substantially from the previous causal
14.4%. Non-causal KV forwarding changes the bottleneck. The refined path has
much higher SFPU activity and much longer execution, exposing compute overhead
that was mostly hidden in the causal case.

### Counter overflow check

The profiler records event counts and reference counts as uint32. The improved
256K operation lasts over 2^32 cycles, so its reference counter wraps once.
The stock per-core ratios consequently report nonsense (e.g. SFPU utilization
over 100%). Its combined MATH event count also wraps; that metric is omitted.

`analyze_counter_timestamps.py` compares reference counts with 64-bit TRISC1
start/end timestamps and restores the reference wrap. It does not silently
unwrap event counts. A second capture at S=131072 lasts less than 2^32 cycles
and verifies those counts: the 256K/128K total-event ratios are 4.00008 for FPU
and 3.99999 for SFPU, consistent with quadratic work and small linear overhead.
The improved path's corrected per-core FPU medians agree: 24.1924% at 256K and
24.1908% at 128K. Full-chip values are 22.08% and 21.96%.

Therefore the reported 256K FPU/SFPU totals have not themselves wrapped. Do
not quote the uncorrected FP32 per-core utilization columns from the raw CSV.

### Timing/telemetry caveat

Single-capture, clock-derived profiler durations were 2369.62 ms (streaming),
2843.61 ms (main FP32), and 5394.32 ms (improved FP32). They are not interchangeable
with the warmed trace-wall medians above. Use the same timing method for ratios.
The profile header reports 1350 MHz. A telemetry snapshot during the H=10 run
reported 150 W power, a 150 W power limit, 1350 MHz AICLK and 79.9 C. BF16 trace
timings drifted with repeated execution. No power/clock settings were changed;
the exact contribution of power management versus capture/run conditions was
not isolated with a clock trace. The large regression is present with either
timing method, but should not be summarized as one configuration-independent
percentage.

## Baseline integrity and reproduction

The exact three-file candidate patch was reverse-applied on the allocated
checkout, `git diff --exit-code` confirmed clean tracked sources, and the host
was rebuilt before main measurements. The candidate was subsequently reapplied,
rebuilt, and checked with `git apply --reverse --check`. Local candidate C++ was
not reverted. No model/kernel optimizations were implemented during this turn.
Both builds used:

```
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
```

Using the remote environment in REPORT.md, the benchmark command is:

```bash
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --kv-lens 262144 --full --heads 10 --variants hifi2 fp32_hifi2 \
  --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 \
  --benchmark-iters 7 --label noncausal-h10 \
  --output experiments/sdpa-l2/hifi2-rounding/noncausal-h10.jsonl
```

For counters, prefix with `python_env/bin/python -m tracy -r -p
--profiler-capture-perf-counters=fpu --check-exit-code -o <profile-directory>`,
then the script and its arguments; omit `--benchmark-iters` for the single-op
capture. The 128K validation changes only `--kv-lens` to 131072.

Raw timing/accuracy: `noncausal-h10-candidate.jsonl`, `noncausal-h10-main.jsonl`,
`noncausal-h4-candidate.jsonl`. Counter CSVs and raw device timestamps are under
`noncausal-h10-profile/reports/2026_09_09_18_49_17/`,
`noncausal-h10-main-profile/reports/2026_09_09_18_52_31/`, and
`noncausal-h10-short-profile/reports/2026_09_09_18_54_45/`.
Build logs: `noncausal-h10-baseline-build.log`, `noncausal-h10-restored-build.log`.

Next optimization work should use this non-causal H=10 benchmark as an
acceptance case, not just the favorable causal measurement.
