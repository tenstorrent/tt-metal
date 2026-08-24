# 03. Per-thread zones inside the real pipeline

*Status: experiment running. This file records the design; results to follow.*

## Question

01 compared threads across *different binaries* — the isolate builds, where the
other threads do only synchronisation. 02 showed counters cannot help. What is
left is to measure each thread's own duration **inside the real `L1_TO_L1`
pipeline**, in the build where the bug exists.

Then: when the pipeline's standard deviation is 2,000 cycles, what is each
thread's?

- **A thread carries the variation** — that thread stalls inside its own measured
  window, and its zone is where to look.
- **All three thread zones are stable while the total is not** — the time is lost
  *between* them, in the handoff. That is where the `MATH_PACK` semwait sits
  (`llk_pack_common.h`).

## Method

No kernel change and no new build — an analysis change only, so the instruction
stream is untouched and the effect should survive.

The profiler already records zones on all three threads during an `L1_TO_L1` run.
`_stats_l1_to_l1` discards everything except unpack `ZONE_START` and pack
`ZONE_END`. Patching it to also emit per-thread durations — the pattern
`_stats_l1_congestion` already uses — yields the pipeline total and each thread's
own duration from a single run.

Then `perf_math_matmul`, `--perf-run-types L1_TO_L1`, `run_count=5`, so every
variant reports `mean` and `std` per thread.

## Why this is better than what came before

| approach | problem |
|---|---|
| Isolate comparison (01) | Different binaries; the other threads do not do real work |
| Hardware counters (02) | The effect is absent from that build |
| **Per-thread zones** | Same binary, same run, effect present |

## Results

*To be filled in.*
