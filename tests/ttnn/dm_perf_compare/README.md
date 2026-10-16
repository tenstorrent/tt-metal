# DM perf compare

Measures, on one machine, whether a change moved the speed of a data-movement op and whether the
default call picks the fastest implementation.

- **Q1: did the change alter speed?** `default` before vs `default` after.
- **Q2: does the default pick the fastest option?** `default` after vs `native` after and `codegen`
  after (`ttnn._ttnn.operations.data_movement.<op>_force_native` / `_force_codegen`).

`default` is the op as users call it, `ttnn.<op>`. A forced `codegen` call that the generated path
refuses is skipped and listed, not counted as an error.

```bash
python tests/ttnn/dm_perf_compare/run.py \
    --calls tests/ttnn/dm_perf_compare/configs/repeat_smoke.json \
    --before-tree /path/to/build-of-merge-base --after-tree /path/to/build-of-change \
    --out results.json
```

Each tree is a built tt-metal checkout; the script runs each side's processes with that tree's
`TT_METAL_HOME` and `PYTHONPATH`. Needs one card; processes run one after another.

## Method

- 5 rounds. A round runs one fresh process per build; their order alternates between rounds.
  The after-build process measures `default`, `native` and `codegen` on every call.
- Per call and implementation: warm-up, then best of N timed calls, each synced. Profiler off.
  `best_us` is host launch plus device, the time a caller waits.
- A call's change is the median of its per-round changes (negative is faster).
- Noise: a second pair of processes runs the after build against itself on a few calls; the noise
  band is the largest per-call median change found there, never below 1%.
- Verdict: `faster` / `slower` only when the median clears the noise band and at least 4 of 5 rounds
  point the same way; otherwise `no change`.
- Output match: every run hashes its output. Runs being compared must agree, otherwise the call is
  `not comparable` (a run that skips work would look fast).
- Device time: a separate pass of the after build under the device profiler. Device time is the sum
  of kernel durations between signposts, best of the reps, median over rounds. The device ops each
  implementation dispatched are recorded, and `default_uses` is whichever of native/codegen has the
  same ops as the default.

## Call file

See `calls.py`. One op, a list of calls (`shape`, `dtype`, `layout`, `memory_config`, `args`).

## Tests

`pytest tests/ttnn/dm_perf_compare/test_dm_perf_compare.py` covers the comparison logic without a device.
