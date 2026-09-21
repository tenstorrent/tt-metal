# Wan block device-performance breakdown

Follow-up: the grid/chunk hypotheses below have now been tested directly.
See [exact-shape SDPA tuning](SDPA_TUNING.md), including sustained timings.

New device-counter measurements on bh-lb-08, IRD 221619, eight Blackhole,
SP4/TP2. Same real high-noise block-20 inputs for all choices: 32,760 valid
tokens, 32,768 padded, 20 local heads, local Q8192, head width 128. Attention
math, formats, masks and block sizes are unchanged from the video suite.

## Measurements

All times are milliseconds. Each entry averages the median of three samples
from each of two opposite-order passes. Full-block round medians differ by
less than 0.08%; all 14 full-block replay checks were bitwise exact on the
checked device output. These are instrumented, isolated trace-replay timings,
not sustained video-generation wall times.

| Choice | Other block work* | Q/KV preprocessing | KV all-gather | SDPA | Full block | Extra vs stock |
|---|---:|---:|---:|---:|---:|---:|
| Stock | 22.02 | 0 | Included in ring | 14.88, including ring | 36.91 | — |
| D | 22.03 | 0 | 6.73 | 40.59 | 69.34 | +32.43 / +87.9% |
| C | 22.06 | 0 | 6.73 | 33.15 | 61.92 | +25.02 / +67.8% |
| B | 22.02 | 0 | 6.73 | 19.18 | 47.91 | +11.00 / +29.8% |
| E | 21.99 | 0.53 | 3.77 | 16.15 | 42.44 | +5.53 / +15.0% |
| F | 21.96 | 0.53 | 3.77 | 20.38 | 46.64 | +9.73 / +26.4% |
| G | 22.01 | 0.49 | 3.73 | 16.12 | 42.35 | +5.44 / +14.8% |

*Other block work is measured full-block duration minus the separately
measured attention replacement path. It includes QKV/output projections,
normalization/RoPE, cross-attention, FFN, residuals and their communication;
these have not been individually decomposed. It is stable at approximately
22 ms. Independently measured stage sums differ slightly from full-path
measurements; they are not forced to add up exactly.

E/F preprocessing is approximately 0.21 ms for Q and 0.32 ms for K+V.
G is approximately 0.21 ms for Q and 0.28 ms for K+V. D/C/B do not run
external preprocessing; their in-kernel numeric work is included in SDPA.
The SDPA column includes the complete kernel, including on-chip/DRAM movement,
softmax and recurrent-state work; it is not FPU-only time.

## What explains the overhead?

- Most overhead is in the changed attention path, not the rest of the block.
  D spends 47.31 ms in that path versus stock's fused 14.88 ms. Approximately
  6.73 ms is explicit gather and 40.59 ms is the standalone SDPA kernel.
- Preprocessing is not a major cost: eliminating it entirely would save only
  about 1.2% of the E/G block, or 1.1% of F. Fusion can help but is not the
  first project to pursue for performance alone.
- F's SDPA kernel is actually slower than B's here: 20.38 versus 19.18 ms.
  F wins overall because compressed-KV gather saves about 2.96 ms, more than
  its extra compute and 0.53 ms preprocessing cost.
- G is essentially tied with E for both compute and communication despite
  much smaller KV. BFP4's compression benefit is not translating into a
  corresponding gather-latency reduction in this implementation.

## Prioritized opportunities

1. **Use the full compute grid in the gather-based adapter.** The hardware
   grid is 12x10. The adapter budgets 12x9=108 cores, then rounds down to a
   whole number per head: five cores/head, 100 active cores. Each head has
   32 Q256 jobs; the busiest cores run seven jobs. Six cores/head would use
   120 cores and reduce the maximum to six jobs. That suggests roughly 14%
   less SDPA time in an ideal job-count model: about 5.8 ms for D, 4.7 ms for
   C, 2.7 ms for B, and 2.3–2.9 ms for E/F/G. These are estimates, not measured
   speedups. The current standalone gather completes before SDPA, so reserving
   compute cores for overlapping CCL appears unnecessary; validate fabric,
   semaphore and L1 placement plus exact outputs before changing it. This
   does not require changing Q/K chunks or input-buffer depth.

2. **Overlap KV movement with attention / integrate the numeric recipes into
   ring attention.** Visible gather costs 6.73 ms for BF16 KV and approximately
   3.75 ms for compressed KV. Perfectly hiding it, with everything else held
   fixed, is a ceiling of roughly 8–14% block-time improvement depending on
   the choice. It is not a guaranteed saving: ring integration changes
   scheduling, local chunk boundaries, online-softmax state merges and
   memory contention, so accuracy must be requalified. D/C still have large
   compute costs even if gather is hidden completely.

3. **Make compressed-KV CCL scale with bytes, especially BFP4.** Source shows
   a default 4352-byte fabric payload and a four-destination scatter-write
   cap in the default all-gather factory. That packs two BF16 tiles, four
   BFP8 tiles, and only four BFP4 tiles per packet, even though seven BFP4
   tiles fit by bytes. E/G therefore have the same tile-packet count under
   those defaults. This is a concrete explanation to test for the measured
   3.77/3.73 ms plateau, not yet an experimentally isolated root cause. The
   cap reflects the scatter command format; increasing the constant alone
   is not a valid fix. Investigate KV-specific layout/packetization, worker
   count, buffering, persistent gather buffers and K/V scheduling. Existing
   hyperparameters depend on shape, not dtype.

4. **Then optimize the remaining SDPA loop and reader.** FP32 choices have
   one K/V CB slot versus two for BF16 choices in this adapter. The shared
   reader also barriers every two DRAM tile reads and serializes chain
   forwarding before publishing each chunk. Measure compute versus unpack/
   pack/data stalls before changing buffering or reader scheduling. These
   are opportunities without deliberately lowering precision; changing
   HiFi/exp/rounding instead would select a different accuracy tradeoff.

5. **Common block optimization is a separate opportunity.** Approximately
   22 ms remains regardless of the attention choice. Matmul logs report
   generic fallback blocking for several Wan shapes. A separate projection/
   FFN/norm/CCL breakdown would determine whether tuning them is worthwhile;
   the current data cannot assign that 22 ms to individual operators.

## Method and caveats

The profiler uses the maximum per-chip first-to-last kernel span for a
component, preserving device dispatch gaps and avoiding double-counting
overlapping ring programs. Profiler timestamps are device cycles; conversion
uses the longest program's reported ns duration divided by its timestamp
delta. Raw per-chip programs, intervals and samples are retained in
[perf-breakdown-02.json](perf-breakdown-02.json). Host trace latency is recorded
as a cross-check. The full-block and attention-path checks agree with their
host trace latency to well below a millisecond.

An initial 100 stock-block warmup replays precedes the sweep; each variant's
block has another 12 warmups, components have five, and each timed replay
is followed by a profiler drain. Profiler drains create pauses, and these
are not sustained thermally steady measurements. Device clocks were not
locked. Reverse-order agreement demonstrates local repeatability, not
absence of instrumentation or duty-cycle bias.

This corrects the earlier pilot-only block comparison, which had five
warmups and different inputs per choice. It also does not replace the
measured video runtimes: the video suite is untraced and includes host
dispatch, expert reloads and sustained-workload effects. E/G being slower
than stock in this isolated traced-block profile but faster in the video
suite is therefore not a contradiction. Determining the exact contribution
of host dispatch versus duty-cycle effects requires matched traced/untraced
whole-model profiling; it has not been isolated here.

The profiler preflight passed. The final sweep passed in 371.36 seconds;
kernel instrumentation was JIT-compiled successfully. The first attempt
failed due to a harness block-selection bug after expert unloading; no data
from that attempt is used. Reproduction source:
[test_perf_breakdown.py](test_perf_breakdown.py). No attention algorithm or
production implementation was changed by this diagnostic pass.
