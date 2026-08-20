# LLK perf gate: cost and threshold baselines

Two architectures, speed of light off, `main`. Details in `blackhole-nonsol/`
and `wormhole-nonsol/`.

Two questions were asked. How long would a gate run take, and how large must a
slowdown be before it means anything.

## Cost

Wall clock on one card, cold build tree.

| config | Blackhole | Wormhole |
|---|--:|--:|
| full — every run type | 19:41 | 30:14 |
| isolates — unpack, math, pack | 12:40 | 18:45 |
| L1_TO_L1 only | 6:17 | 9:19 |

The two architectures run on separate CI runners in parallel, so gate latency is
the Wormhole figure: about **9 minutes** for L1_TO_L1, **19 minutes** for
isolates.

Compile is 65-79% of every configuration, so build caching on the gate runner
would save more than any run-type choice.

Cost tracks the number of (module x run-type) pairs almost linearly and the
proportions match across architectures, so any future configuration's cost is
predictable from the pair count.

## Threshold

We ran identical code five times on one card and measured how much each number
moved. Nothing changed between runs, so every difference is noise.

> **Flag a regression when a number is more than 2% slower AND more than 30
> cycles slower.**

| arch | configuration | measurements | rule fires on unchanged code |
|---|---|--:|--:|
| Blackhole | L1_TO_L1 | 108,377 | **0** |
| Blackhole | MATH_ISOLATE | 101,180 | **0** |
| Blackhole | UNPACK_ISOLATE | 103,424 | **0** |
| Blackhole | PACK_ISOLATE | 106,748 | 2 |
| Wormhole | L1_TO_L1 | 100,971 | 53 |
| Wormhole | MATH_ISOLATE | 94,692 | **0** |
| Wormhole | UNPACK_ISOLATE | 96,846 | **0** |
| Wormhole | PACK_ISOLATE | 99,414 | 1,457 |

Both clauses are necessary. A percentage alone fails on INIT, which is a few
hundred cycles and wobbles by 20; a cycle count alone fails on TILE_LOOP, which
moves thousands of cycles and is still under 2%.

**Repeating runs does not help.** Averaging two runs per side does not improve
the typical case at all, only the extreme tail. Use one run per side.

## What is gate-ready today

| configuration | verdict |
|---|---|
| MATH_ISOLATE, UNPACK_ISOLATE | **Ready.** Zero false positives on both architectures |
| L1_TO_L1 | Ready on Blackhole. On Wormhole it carries 53 matmul points |
| PACK_ISOLATE | **Not ready.** See below |

## The one blocker: the packer

Every failure in the study is the packer path.

- Wormhole PACK_ISOLATE: 1,457 failures, up to 24%. 1,246 are `perf_matmul`.
- Wormhole L1_TO_L1: 53 failures, all matmul.
- Blackhole PACK_ISOLATE: 2 failures — one measurement seen through two markers,
  so a single point. Nothing can be inferred from it.

What the characterisation shows on Wormhole (`wormhole-nonsol/outliers_pack_isolate.md`):

- **Each measurement is affected independently.** The run that disagrees is
  spread 24 / 23 / 20 / 15 / 17 percent across the five, against 20% under
  independence. There is no per-run state.
- **The deviation is symmetric** — mean +0.6%, from -19.8% to +24.3%. The odd
  run is as often faster as slower, so nothing is being added to some runs; the
  measurement lands in one of two places.
- **76% are exactly two-valued** across the five runs, consistent with a
  discrete state change for most of them.
- **Output format separates them.** Within `perf_matmul`, Float16_b output has a
  13% flag rate against 0% for Float32 and Bfp8_b.

No usable threshold can absorb a 24% swing, so this needs an investigation
before the packer can be gated.

## Open questions

1. **Why does Float16_b output destabilise the packer?** The association is
   strong and the mechanism is unknown. This is the concrete lead. The single
   Blackhole offender also writes Float16_b, which is consistent, though one
   point proves nothing.
2. **Are the 53 L1_TO_L1 failures the same phenomenon?** They do not share the
   signature: 21% two-valued against 76%, deviations ±4.6% against ±24%.
3. **How big are real regressions?** These numbers constrain the threshold from
   below only. The upper bound has to come from the commit history.
4. **Cross-machine and cross-day drift.** Every run here was one card in one
   session. A real gate compares different runners on different days.

Cross-core contention was an early hypothesis — 15 tests measure concurrently,
and the affected points are large. It is now unlikely: contention would not
depend on the kernel's output format, and it would not produce a two-valued
distribution. Measuring serially would still settle it, but it is no longer the
leading explanation.
