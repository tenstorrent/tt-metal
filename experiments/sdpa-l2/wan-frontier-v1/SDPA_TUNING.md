# Wan exact-shape SDPA: measured grid, chunk and reader controls

2026-09-16, bh-lb-08, IRD 221619, eight Blackhole p150b devices.

## Conclusion

Grid/chunk choices really did cost performance, but they were not a complete
explanation for every recipe. G gains substantially from both fixes. E gains
from the grid fix but not meaningfully from the tested chunk/reader changes.
B/C/D still cost more than stock for this shape; their benefit is accuracy.

Against **tuned stock**, isolated device timings show E 1.05x and G 1.26x
speedup. Two uninstrumented sustained passes show E 1.21–1.34x and G
1.46–1.64x speedup. Clock/power behavior changes the relative results, so the
two timing regimes must not be mixed. These are pure SDPA results, not new
block or video-generation measurements.

## Exact workload and fair baseline

- Real high-noise Wan2.2 block-20 tensors, captured from the same stock pilot.
- SP4/TP2 on a 2x4 mesh: 40 global heads, 20 heads per device, head width 128.
- Per-device Q `[1,20,8192,128]`; gathered K/V `[1,20,32768,128]`.
- Non-causal, 32,760 valid keys. All candidates mask the final eight keys.
- The complete SDPA kernel is timed, including real DRAM reads, on-chip KV
  forwarding, softmax, recurrent state and output writes. This is **not** a
  data-movement-ablated or L1-resident microbenchmark.
- Preparation and KV all-gather are outside the timed region for **all**
  choices. E/F/G are prepared on the owning SP rank before gathering.
- Stock uses the model's ordinary BF16/HiFi2 configuration, accurate-exp
  setting, and the regular rectangular SDPA API. K/V have logical length
  32,760 and physical length 32,768, selecting stock's native padding mask.
  No dense/windowed mask or extra mask-reading traffic is introduced.
- Stock's earlier **14.88 ms ring-attention timing is not this baseline**:
  it includes a different, fused ring/compute execution path. Here stock and
  the candidates all consume already-gathered KV.
- A is the frozen older-main recipe in the private adapter, not identical
  to current stock's configuration. F is retained as an auxiliary comparison
  even though the current selected set is D/C/B/E/G.

## Best measured settings

All rows use 120 active compute cores. “Best” means the best configuration
from this bounded sweep, not a proof of a global optimum. Input-buffer depths
and numerical recipes are unchanged: FP32 recipes keep one K/V slot, BF16
recipes keep two, and Q keeps its original buffering.

| Variant | Q/K chunks | Intermediate read barrier | Isolated device ms | Sustained ms, two passes | Sustained effective TFLOP/s/chip |
|---|---:|---:|---:|---:|---:|
| Stock | 512/256 | Stock reader | 14.47 | 20.43–21.71 | 126.6–134.5 |
| D: accurate FP32 | 128/1024 | End of chunk only | 28.67 | 42.47–43.05 | 63.8–64.7 |
| C: FP32 QK4/PV2 | 256/512 | Every 2 tiles | 28.41 | 35.94–36.15 | 76.0–76.5 |
| B: compensated BF16 | 256/512 | Every 8 tiles | 16.38 | 22.47–23.25 | 118.2–122.3 |
| E: LoFi, BFP8 KV | 256/512 | Every 8 tiles | 13.83 | 16.21–16.83 | 163.3–169.5 |
| G: LoFi, BFP4 KV | 256/1024 | Every 8 tiles | 11.44 | 13.20–13.99 | 196.4–208.2 |
| A: frozen main recipe | 512/256 | Every 2 tiles | 14.45 | 20.53–21.41 | 128.4–133.9 |
| F: FP32 LoFi, BFP8 KV | 256/512 | Every 8 tiles | 17.47 | 19.72–20.43 | 134.5–139.3 |

Effective FLOPs count the two attention matmuls:
`4 * 20 * 8192 * 32760 * 128 = 2.7481079808e12` per device invocation.
They are not an FPU-busy measurement or a utilization percentage against a
common peak: different fidelities have different hardware throughput limits.

With the model's original Q256/K256 chunks, standalone stock measures
16.54 ms isolated and 23.38–23.54 ms sustained. Stock was therefore tuned too,
not left at its model defaults to inflate candidate speedups.

## Controlled attribution: what actually helped?

Each entry below is an isolated **device-counter** median in milliseconds.
The same inputs and numerical recipes are used throughout. The baseline is
Q256/K512; stock's original-model Q256/K256 configuration is described above.

| Variant | Original 108-core budget | Full grid, same chunks | Best chunks, same reader | Best tested reader too |
|---|---:|---:|---:|---:|
| Stock | 17.87 | 16.27 | 14.47 | 14.47 |
| A | 18.23 | 16.35 | 14.45 | 14.45 |
| D | 40.59 | 34.81 | 30.42 | 28.67 |
| C | 33.15 | 28.41 | 28.41 | 28.41 |
| B | 19.19 | 16.75 | 16.75 | 16.38 |
| E | 16.15 | 13.85 | 13.85 | 13.83 |
| F | 20.38 | 17.48 | 17.48 | 17.47 |
| G | 16.12 | 13.83 | 11.48 | 11.44 |

1. **Grid penalty proven.** The private adapter's 108-core budget rounds
   down to five cores/head, only **100 active cores**. Q256 gives 32 jobs/head,
   distributed 7/7/6/6/6. The full grid gives six cores/head, **120 active
   cores**, with 6/6/5/5/5/5 jobs. Changing only this setting cuts D/C/E/F/G
   time by about 14%, B by about 13%. The sampled outputs are bitwise exact.
   Stock's global scheduler is different and can use its full 108-core budget.

2. **Chunk penalty proven for D and G, not uniformly.** D prefers Q128/K1024
   in the tested set. G can fit Q256/K1024 because its KV buffers are smaller;
   this reduces its K-loop iterations from 64 to 32. At fixed full grid and
   the original reader, G drops 13.83 → 11.48 ms. E's larger BFP8 buffers
   prevent that configuration under unchanged buffering; its best tested
   chunks remain Q256/K512. Smaller K blocks generally increase the
   recurrent-state/softmax overhead, and smaller Q blocks often hurt.

3. **The reader-barrier hypothesis was mostly disproven.** At the chosen
   chunks, going from barriers every two reads to every eight/end-of-chunk
   helps D by about 5.8% and B by about 2.2%. E/F/G gains are under 0.4%,
   too small to call a meaningful remaining bottleneck here. The final
   end-of-chunk read barrier is never removed. No chain algorithm or
   data layout is changed. Reader controls preserve sampled outputs exactly.

4. **The grid gain survives sustained execution, but clock changes reduce
   it.** An additional matched, tuned-chunk test in the reverse pass gives
   D 45.34 ms at 100 cores versus 43.05 ms at 120, and G 14.97 versus
   13.20 ms. Thus the result is not only a cool-chip profiler artifact,
   but the isolated 14% reduction is not a universal sustained prediction.

## Accuracy did not disappear in the tuning

Reference: FP64 attention on the original BF16 inputs, all 20 heads and
32 evenly spaced query rows on the first device, attending to all 32,760
valid keys. This is 640 sampled query rows from a real block, **not** the
full qualification suite or an assertion about all Wan layers/prompts.

| Variant | Tuned L2 % | Tuned PCC |
|---|---:|---:|
| Stock | 2.4150 | 0.999765120 |
| A | 2.2977 | 0.999767764 |
| D | 0.1664 | 0.999998615 |
| C | 0.1931 | 0.999998216 |
| B | 1.1791 | 0.999936938 |
| E | 1.1393 | 0.999935202 |
| F | 0.8485 | 0.999965441 |
| G | 5.6385 | 0.998410236 |

G's Q256/K512 L2 was 5.6466%; Q256/K1024 is 5.6385%, essentially the same
band. D remains about 0.1664% throughout. Stock/A change accumulation error
with K chunk size: the faster K256 setting has higher error than K512.
Their selected K256 setting matches the model's original K chunk size.
Input formats remain Q=BF16 for every choice; K/V=BF16 for A/B/C/D,
BFP8_B for E/F, and BFP4_B for G. The same Q-rounding and KV-quantization
preprocessing from the pinned recipes is used, not a new numerical shortcut.

## Why two timing regimes are necessary

The isolated sweep uses 12 warmup trace replays and five counter samples,
draining profiling data between replays. It takes the maximum per-chip
kernel span over all eight chips. This is consistent with the earlier block
breakdown and is useful for controlled grid/chunk comparisons.

Ten-queued-replay checks exposed changing clock conversion on FP32 runs.
Consequently the sustained results above come from a **separate process
with device profiling disabled**, not the drifting profiler ns conversion.
They use synchronized wall time amortized over 20 queued trace replays,
five samples per point, after 100 warmups in the forward pass and 200 in
the reverse pass. They include dispatch/synchronization overhead. The table
ranges are the two pass medians, not confidence intervals or sample extrema.
The per-row `perf.warmup` fields are authoritative: the reverse JSON's
top-level descriptive string retained the initial 100-warmup wording,
although its measurements used 200. The harness description is now corrected.

Live telemetry during the reverse pass recorded AI clocks of 1200–1231 MHz
in one snapshot and 950–1018 MHz in another, versus a configured maximum of
1350 MHz. The latter snapshot has chip 0 at 68.8 C, about 145 W, with a
150 W configured TDP limit and a 90 C first thermal limit. This is consistent
with power/current-related clock limiting; the precise control mechanism
was not separately isolated. No clock/power/fan settings were changed.
Repeat ordering and clock state matter: do not combine the lowest candidate
timing from one regime with a stock baseline from another.

## Scope, verification and reproducibility

- 78 distinct successful configuration points, 96 instrumented measurement
  rows including rechecks; 28 uninstrumented rows across the two passes.
- Chunk candidates: Q/K 128/256, 128/512, 128/1024, 256/128, 256/256,
  256/512, 256/1024, 512/128 and 512/256. The last-added 256/1024 case was
  run for G; other recipes exceed the unchanged-buffer capacity guard.
- Near-capacity configurations are not silently substituted. Some are
  skipped by the CB-size guard; D/C at 512/128 and G at 512/256 were rejected
  by the runtime because their CBs clash with existing L1 allocations.
- The first sweep stopped when stock 128/1024 exceeded L1. The continuation
  preserved completed results, verified identical first-device Q/K/V, and
  completed successfully. No invalid configuration contributes a timing.
- Device-kernel changes are limited to parameterizing chunk geometry; the
  changed kernels were JIT-compiled and executed in the sweep. Numeric
  headers, fidelity, destination precision, preprocessing and input-buffer
  depths are unchanged. The model adapter's default choices remain unchanged
  so prior videos remain reproducible; tuning knobs are opt-in.
- Python syntax checks and Black formatting were run. The initial sweep's
  expected L1 rejection is retained in its log, rather than erased.

Test: [test_sdpa_tuning.py](test_sdpa_tuning.py).
Device input adapter: [device_attention.py](../flux2-frontier-v1/device_attention.py).

Artifacts:

- [Counter sweep](sdpa-tuning-01.json), [initial log](sdpa-tuning-01.log),
  [continuation log](sdpa-tuning-resume-01.log), [G extension log](sdpa-tuning-g-01.log).
- [Sustained forward](sdpa-wall-01.json), [forward log](sdpa-wall-01.log).
- [Sustained reverse](sdpa-wall-02.json), [reverse log](sdpa-wall-02.log).
- [Active telemetry 1](sdpa-wall-02-telemetry-01.json),
  [active telemetry 2](sdpa-wall-02-telemetry-02.json).

On the reserved container, using the established Wan environment and cache:

```bash
# Device-counter sweep (new output path required).
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \
WAN_SDPA_REPORT=/absolute/path/sweep.json \
/opt/venv/bin/python -m pytest experiments/sdpa-l2/wan-frontier-v1/test_sdpa_tuning.py -s -x -q

# Uninstrumented verification, choosing settings from that completed sweep.
TT_METAL_DEVICE_PROFILER=0 \
WAN_SDPA_TIMING_ONLY_SOURCE=/absolute/path/sweep.json \
WAN_SDPA_REPORT=/absolute/path/wall.json \
/opt/venv/bin/python -m pytest experiments/sdpa-l2/wan-frontier-v1/test_sdpa_tuning.py -s -x -q
```

`WAN_SDPA_RESUME=1` resumes a sweep without discarding results.
`WAN_SDPA_BEST_ONLY=1`, `WAN_SDPA_WARMUP=200`,
`WAN_SDPA_VARIANTS='G F E B C D A stock'` and
`WAN_SDPA_TUNED_GRID_CHECK=1` reproduce the reverse-pass choices.
Weight conversion fallback remains forbidden. Captured full first-device
inputs remain on the reserved machine alongside these reports as
`sdpa-tuning-01.inputs.pt`; model weights were loaded from the existing cache.

## Implications

Use full-grid scheduling and recipe-specific chunks when integrating these
choices. G has a real pure-op speed advantage, E a smaller but repeatable one.
B/C/D should be justified by their measured accuracy rather than a presumed
speedup over stock. F's sustained behavior is better than its isolated
comparison suggests and should not be dismissed on isolated timing alone.

This does not measure tuned full-block or video speedup. Explicit KV gather
and preparation still cost time in the private adapter, while stock can
overlap communication in its ring path. Applying the numeric recipes to
that path, and evaluating the result under actual model duty cycles, is a
separate integration step.
