# 02. Hardware counters do not reproduce the effect

## Question

Timing cannot separate a thread that is *waiting* from one that is *working*.
Hardware counters can. Which counter goes bimodal when the measurement does?

## Method

Every perf test source is compiled twice from the same file. `START_PERF_MEASURE`
expands to `ZONE_SCOPED` in the NC build and to `MEASURE_PERF_COUNTERS` in the WC
build — never both, so wall-clock and counters are never collected together.

Run `perf_math_matmul` under `L1_TO_L1` with `run_count=5` twice: once in the NC
build to learn which configurations flip, once in the WC build for counters. Join
on configuration.

19,920 variants; 25 flagged by the timing pass.

## Result

Standard deviation across the five repeats, same configurations, both builds:

| | clean configurations | flagged configurations |
|---|--:|--:|
| **NC build** (timing), median | 3.8 | **2,034.6** |
| NC build, min / max among flagged | — | 704 / 4,573 |
| **WC build** (counters), median | 1.8 | **2.2** |
| WC build, exceeding 100 cycles | 258 of 19,895 | 4 of 25 |

Mean zone duration is unchanged between builds: **297,286 versus 298,241
cycles**.

## Conclusion

In the timing build the flagged configurations stand out by a factor of 500. In
the counter build they are indistinguishable from everything else, while
measuring the same amount of work.

**The counter build does not reproduce the bug. That instrument is closed.**

## Why this is still a result

The total work is identical and only the instrumentation instruction stream
differs, yet the effect disappears. That is the signature of a **race with a
window narrow enough that a few instructions of profiling code close it**. A
fixed data-dependent cost — a slower conversion path, an extra pass — would
survive the change. This does not.

## A misreading to avoid repeating

An earlier pass ranked counters by whether their `std` was non-zero more often on
flagged configurations than on clean ones, and produced an apparently strong
result led by `INSTRN_THREAD.SYNC_INSTRN_AVAILABLE_2` and
`WAITING_FOR_NONZERO_SEM_2` — the packer's semaphore wait. **It was wrong**, for
two reasons.

1. The variations being ranked were single-digit counts, not the phenomenon.
2. The follow-up used the `.cycles` columns as if each were cycles attributable
   to that counter. Per the counters guide, every bank reports `OUT_L`, the
   **total elapsed cycles of the zone**, alongside its selected event count. The
   `.cycles` suffix is `OUT_L`, so it is the same value for every counter — which
   is why five unrelated counters returned identical figures on all 25 rows.

The event counts are the un-suffixed columns. But since the effect is absent from
the WC build entirely, no counter analysis can reach it.
