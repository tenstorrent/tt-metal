# DM perf compare

Measures, on one machine, whether a change moved the speed of a data-movement op and whether the
default call picks the fastest implementation.

- **Q1: did the change alter speed?** `default` before vs `default` after.
- **Q2: does the default pick the fastest option?** Device time of `default` after vs `native` after
  and `codegen` after (`ttnn._ttnn.operations.data_movement.<op>_force_native` / `_force_codegen`),
  plus `default_uses`, the implementation the default dispatched. Host time for the same pairs is
  kept as `q2_host` for detail only: the default's public entry and routing check cost a few
  microseconds the forced entries skip, so it can never come out fastest on host time.

`default` is the op as users call it, `ttnn.<op>`. A forced `codegen` call that the generated path
refuses is skipped and listed, not counted as an error.

```bash
python tests/ttnn/dm_perf_compare/run.py \
    --calls tests/ttnn/dm_perf_compare/configs/repeat_smoke.json \
    --before-tree /path/to/build-of-merge-base --after-tree /path/to/build-of-change \
    --out results.json
```

`--op <op>` instead of `--calls` uses the default list, `configs/sweeps/<op>.json`: 40 calls drawn
with a fixed seed (0) from the op's sweep grids. Regenerate with `sample_sweeps.py`; it needs no
device. A change that brings its own calls passes them with `--calls`.

Each tree is a built tt-metal checkout; the script runs each side's processes with that tree's
`TT_METAL_HOME` and `PYTHONPATH`. Needs one card; processes run one after another.

## Method

- 5 rounds, after one discarded priming pair. A round runs fresh processes: before (default only),
  after (default only), then after again with `default`, `native`, `codegen` and `default_b` (the
  default measured a second time: Q2's own A-vs-A control, reported as `q2_control`). The first two
  feed Q1 and alternate order between rounds; the third feeds `q2_host` and the control. Q1 compares like with like
  because a process that also runs the forced entries leaves caches and allocator in another state
  (measured: about +4% on the same default call).
- Per call and implementation: warm-up, then best of N timed calls, each synced. Profiler off.
  `best_us` is host launch plus device, the time a caller waits.
- A call's change is the median of its per-round changes (negative is faster).
- Noise: a second pair of processes runs the after build against itself on a few calls; the noise
  band is the largest per-call median change found there, never below 3%.
- A noise band above 5% means the control pair was disturbed (host busy in some rounds): the
  results file marks `noise.reliable: false` and the run should be repeated, since "no change" and
  "default is fastest" from such a run prove nothing.
- Verdict: `faster` / `slower` only when the median clears the noise band and at least 4 of 5 rounds
  point the same way; otherwise `no change`.
- Output match: every run hashes its output. Runs being compared must agree, otherwise the call is
  `not comparable` (a run that skips work would look fast).
- Device time: a separate pass of the after build under the device profiler. Device time is the sum
  of kernel durations between signposts, best of the reps, median over rounds. The device ops each
  implementation dispatched are recorded, and `default_uses` is whichever of native/codegen has the
  same ops as the default. Q2's verdict compares these per-round device times against the same noise
  band as host time. Both sides of Q1 and the device pass run on profiler-enabled builds, so the host
  times carry the same few-percent profiler overhead on each side.

## Call file

See `calls.py`. One op, a list of calls (`shape`, `dtype`, `layout`, `memory_config`, `args`).

## Tests

`pytest tests/ttnn/dm_perf_compare/test_dm_perf_compare.py` covers the comparison logic without a device.
