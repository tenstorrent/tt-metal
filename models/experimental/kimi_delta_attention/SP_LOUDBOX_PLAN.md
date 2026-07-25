# KDA LoudBox sequence-parallel prototype

## Objective

Turn the existing `TP=8` whole-head KDA implementation into a correctness-first
sequence-parallel prototype on the available eight Blackhole devices, without
regressing the existing path.  The production scale-out shape remains
`SP=8 x TP=4` on 32 Galaxy devices; LoudBox is the place to validate the
state-transfer protocol and measure its components.

## Baseline (verified on this branch)

* Branch: `pjosipovic/kda-loudbox-sp-prototype`, based on Momcilo's
  `mvasilijevic/codex/kimi-linear-kda` at `671b0c9acb6`.
* Hardware: one eight-device Blackhole LoudBox.
* `test_ttnn_layer.py`: 7 passed (T=1, 4, 32, state continuity and external
  state tests).
* `test_tp_weights.py`: 2 passed; TP=8 output PCC 0.999965, recurrent-state
  PCC 0.999910, convolution-state PCC 0.999997.
* The traced LB TP=8 control at T=5120 completed in 3.114 ms on the slowest
  device (three warm trace replays; report
  `2026_07_25_16_17_42`).  Historical branch reports measured 619.594 us at
  T=640 and 3183.263 us at T=5120.  At T=5120 the output projection's fused
  matmul/reduce-scatter is about 1.04 ms, so its CCL is a primary reason that
  merely tuning local KDA will not meet the next goal.

## Current implementation and measurements (2026-07-25)

* SP=2/TP=4 is functional at the production-rank-equivalent shape: global
  T=5120, H=2304, 32 heads, 8 heads/rank.  Output, recurrent cache, short-conv
  cache, and the first post-boundary token pass PCC >= 0.98.
* TP=4 needs a 96-tile causal-convolution input while the original direct
  kernel holds all channel tiles in each worker's L1.  The kernel now splits
  that into two 48-tile channel blocks; the target PCC test validates the
  split convolution through the full layer.
* The TP=4 long path uses explicit post-scan KDA RMS normalization.  The
  TP=8-only fused scan/RMS variant produced incorrect TP=4 long outputs even
  though the recurrent cache was correct.
* The warmed three-replay trace at global T=5120 is **5.707 ms/forward**
  (report `2026_07_25_18_16_20`), versus the <= **2.958 ms** primary gate.
  This prototype therefore misses the gate by 2.749 ms (93%).  The comparable
  eager signpost is 5.961 ms/forward.  The experimental local
  `QWEN_KDA_GROUP_PREFIX=1` route is correct but slower, 13.197 ms/forward;
  it is not an optimisation to enable.
* The remaining critical-path cause is architectural: first-span recurrence,
  cache transfer, and second-span recurrence are ordered.  Transport tuning
  cannot recover the 2.749 ms deficit.  The next implementation target is
  Milestone 4's inter-span affine scan, which makes the large span scans
  concurrent after a small log-depth prefix over `(A, B)` summaries.
* An opt-in `KDA_SP_AFFINE=1` SP=2 prototype validates the affine state
  transition and overlaps prepared final scans.  Its direct state-only
  span-summary kernel was correct but measured **6.821 ms/forward** (three
  repetitions; report `2026_07_25_18_36_33`).  Replacing it with eight-chunk
  summaries plus the device `KdaAffinePrefixOperation` preserves the full
  T=5120 PCC gate, but is **7.572 ms/forward** (report
  `2026_07_25_18_44_53`): it still computes a summary scan and a final output
  scan.  Both remain disabled by default.
* An opt-in `KDA_SP_PIPELINED=1` control overlaps span two's convolution/input
  preparation with span one's normal recurrence, then transfers the ordinary
  final recurrent state.  It passes the same PCC gate, but measures **7.050
  ms/forward** (report `2026_07_25_18_46_43`), so host scheduling alone is
  also not the lever.
* The concrete next target is a split-phase grouped KDA interface: retain the
  existing `chunk_gdn_prep` results, emit group `(A, B)` summaries and their
  affine prefix, send the prefix-derived final state immediately, then run
  both spans' group output scans from those already-computed entry states.  It
  must eliminate the duplicate summary prep/scan before this topology can
  plausibly reach the 2.958 ms LB gate.
* That split-phase interface is now implemented behind
  `KDA_SP_SPLIT_AFFINE=1`.  Its reusable preparation, summary, prefix, and
  seeded grouped-output scan agree with the regular KDA path (primitive PCC
  >= 0.999; end-to-end SP PCC >= 0.98).  The first three-repetition eager
  measurement is **7.771 ms/forward** (report `2026_07_25_18_55_46`), still
  above the serial control.  Thus removing duplicate preparation alone is not
  sufficient: the next investigation must make the two TP=4 final scans
  genuinely concurrent at the queue/fabric level, rather than only enqueueing
  their work from the host in that order.

## Milestones

1. **Protocol reference and tests.** Add a partitioned PyTorch reference that
   exactly composes KDA spans.  It must show that each sequence boundary needs
   the incoming recurrent state plus the preceding three projected Q/K/V
   samples for the causal short convolution.
2. **LoudBox SP=2, TP=4 topology.** This is the primary end-to-end benchmark.
   Refactor logical TP from physical mesh size, preserve TP=4 output
   reduce-scatter inside each group, and transfer the per-TP-rank carry at the
   single SP boundary.  Use global T=1280, so every rank owns 8 heads x 640
   tokens: the same KDA work per device as Galaxy SP=8, TP=4 at global T=5120.
   Validate output and state PCC >= 0.98.
3. **LoudBox SP=8, TP=1 protocol probe.** Each chip owns one contiguous T/8
   span and all 32 heads.  It validates seven ordered boundaries at T=5120 but
   has four times the target head work and no TP collective, so it is a
   carry/scan correctness and latency probe rather than an end-to-end
   production-performance comparison.
4. **Galaxy-ready affine scan.** Replace serialized state relay by a
   log-depth scan over each span's affine summary `(A, B)`.  For K=V=128 and
   TP=4, a rank owns eight heads: `A` is 512 KiB and `B` is 512 KiB, so one
   uncompressed summary is 1 MiB.  Keep the actual recurrent state FP32; do
   not silently quantize it.
5. **Profiler gate.** Add/profile harnesses for TP=8, SP=8/TP=1 and
   SP=2/TP=4.  Record operation timing and transfer payloads before proceeding
   to Galaxy integration.

## LoudBox performance goals and decision gates

The real LB topology is `SP=2 x TP=4`.  `SP=8 x TP=4` requires 32 physical
devices; LB cannot execute its CCLs as a 32-rank mesh.  A TP=4 group at T=640
does, however, execute precisely the work of one production rank.  This gives
us a direct local-work control and SP=2 supplies the real cross-device
boundary.

* **Controls to record:** trace TP=4/SP=1 at T=1280 and TP=4/SP=1 at T=640,
  then trace SP=2/TP=4 at global T=1280.  All measurements use the same
  warmed trace harness and report the slowest device's layer critical path.
* **Primary LB speed gate:** SP=2/TP=4 at global T=1280 must be at least
  **1.5x faster** than TP=4/SP=1 at T=1280.  The stretch target is **1.7x**.
  This is an end-to-end layer comparison with the same model, global sequence,
  and output semantics; it is the closest full-system experiment LB can run.
* **Head-to-head LB topology gate:** at global T=5120, SP=2/TP=4 must beat
  the traced TP=8 control of 3.114 ms by at least **5%**, i.e. <= **2.958 ms**
  on the slowest device; the stretch target is 10%, <= **2.803 ms**.  Both
  layouts do 20,480 head-tokens per chip, so this is deliberately a stringent
  topology-and-CCL efficiency goal, not a claimed work-reduction speedup.
* **Production-rank budget:** SP=2/TP=4 at T=1280 must be no more than
  **1.25x** the latency of the TP=4/SP=1 T=640 local-work control.  The
  difference is the measurable cost of one real SP boundary plus scheduling;
  it prevents a nominal 2-way speedup hiding an excessive handoff tax.
* **Carry budget:** the 512 KiB FP32 recurrent carry plus 18 KiB BF16 short
  convolution carry, sent per TP rank, must consume <= **10%** of the local
  T=640 layer critical path.  The trace must show no host tensor conversion or
  host-mediated copy.
* **SP=8/TP=1 boundary probe:** seven-hop relay at T=5120 must preserve PCC
  >= 0.98 and expose per-hop latency.  It has no speedup target because its
  per-device head count differs from production.
* **Stop/adjust condition:** if the boundary cannot meet the 10% carry budget,
  do not proceed with a full 32-device integration; first fuse summary
  generation/consumption or reduce the transport representation with an
  accuracy study.

## Reproducible LoudBox gate commands

Run these only after the eight-device board is healthy.  Each profiler command
uses three warmed trace replays by default and brackets the measured region
with a distinct Tracy signpost, so the slowest-device critical path can be
read from the resulting report.

```bash
# Functional target gate: output, recurrent carry, short-conv carry, and the
# first output token after the SP boundary must all reach PCC >= 0.98.
KDA_SP_TARGET_SHAPE=1 scripts/run_safe_pytest.sh -svv models/experimental/kimi_delta_attention/tests/test_sp2_tp4.py

# Local TP=4 controls: run both lengths with the same warmed trace procedure.
PERF_SEQ=640  PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp4_layer_perf.py
PERF_SEQ=1280 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp4_layer_perf.py

# Primary SP experiment and the direct TP=8 topology comparison at global T=5120.
PERF_SEQ=1280 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py
PERF_SEQ=5120 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_sp2_tp4_layer_perf.py
PERF_SEQ=5120 PERF_TRACE=1 scripts/run_safe_pytest.sh --profile -svv models/experimental/kimi_delta_attention/tests/perf/test_kda_tp_layer_perf.py
```

The final two reports are the head-to-head decision: SP=2/TP=4 must be no
slower than 2.958 ms on the slowest device to pass, against the existing
3.114 ms TP=8 control.

## Non-goals for this first slice

* No FP32 recurrent-state downgrade.
* No all-reduce used in place of the ordered causal carry.
* No unrelated TP=8 micro-tuning before the cross-device SP bottleneck is
  measured.
