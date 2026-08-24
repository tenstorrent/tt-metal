# 03. Per-thread zones inside the real pipeline

## Question

01 compared threads across *different binaries* — the isolate builds, where the
other threads do only synchronisation. 02 showed counters cannot help. What was
left was to measure each thread's own duration **inside the real `L1_TO_L1`
pipeline**, in the build where the bug exists.

When the pipeline's standard deviation is 2,000 cycles, what is each thread's?

## Method

No kernel change and no new build — an analysis change only, so the instruction
stream is untouched.

The profiler already records zones on all three threads during an `L1_TO_L1` run.
`_stats_l1_to_l1` discards everything except unpack `ZONE_START` and pack
`ZONE_END`. Patching it to also emit per-thread durations — the pattern
`_stats_l1_congestion` already uses — yields the pipeline total and each thread's
own duration from a single run.

`perf_math_matmul`, `--perf-run-types L1_TO_L1`, `run_count=5`. 19,920 variants,
29 flagged.

## Result

**The effect reproduced.** Standard deviations up to 4,573 cycles, unlike the
counter build in 02. So the analysis-only patch did not perturb it.

Every flagged variant shows all four standard deviations equal to within a few
cycles:

| pipeline | unpack | math | pack |
|--:|--:|--:|--:|
| 4,573 | 4,569 | 4,580 | 4,571 |
| 3,162 | 3,162 | 3,162 | 3,162 |
| 2,112 | 2,108 | 2,108 | 2,112 |

Median share of the pipeline's variation carried by the largest thread: **100%**.

## Why that is not the finding it looks like

The mean zone durations explain it:

| zone | share of the pipeline total |
|---|--:|
| unpack | 99.9% |
| math | 100.1% |
| pack | 100.0% |

**All three zones span essentially the whole pipeline window.** Each thread's
`TILE_LOOP` zone covers its entire loop *including the time it spends waiting on
the other threads*, so when the pipeline stretches, all three stretch with it by
construction.

## Conclusion

The zones are too coarse to localise anything. This measurement cannot
distinguish "the packer stalls and the others back up behind it" from any other
ordering, because all three zones contain the whole event either way.

What it does establish:

- The effect survives an analysis-only instrumentation change, unlike the counter
  build (02). The two builds differ, and the WC build's immunity in 02 is a
  property of that build rather than of measurement in general.
- No thread's zone is *longer* than the others when the pipeline is slow. There
  is no "one thread ran long" signature to find.

## What would be needed next

A zone around a single handshake — for example the `MATH_PACK` semwait in
`llk_pack_common.h` — rather than around the whole loop. That is a kernel change
and it adds instructions to the exact region under suspicion. Given 02 showed the
effect is sensitive enough to instrumentation that swapping the profiling macro
removes it entirely, a finer zone may well remove it too.

That is the limit of what black-box timing can reach.
