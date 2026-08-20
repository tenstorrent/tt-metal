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

- Wormhole PACK_ISOLATE: 1,457 failures, up to 24% and 41,091 cycles. 1,246 of
  them are `perf_matmul`.
- Wormhole L1_TO_L1: 53 failures, all matmul. That path runs unpack, math and
  pack in sequence, so the instability leaks in.
- Blackhole PACK_ISOLATE: 2 failures, one test configuration
  (`perf_pack_dest_bank`), bimodal between the first run and the rest.

On Wormhole it is not a first-run artefact: any of the five runs can be the odd
one out. It is genuine run-to-run instability.

No usable threshold can absorb a 24% swing, so this needs an investigation
before the packer can be gated.

## Open questions

1. **Why is the packer unstable, and worse on Wormhole?** One untested
   hypothesis: the measure phase runs 15 tests concurrently on different Tensix
   cores, so a long kernel overlaps with its neighbours and sees whatever L1 and
   NoC contention occurs. The affected points are the largest ones. Measuring
   serially would confirm or eliminate it.
2. **How big are real regressions?** These numbers constrain the threshold from
   below only. The upper bound has to come from the commit history.
3. **Cross-machine and cross-day drift.** Every run here was one card in one
   session. A real gate compares different runners on different days.
