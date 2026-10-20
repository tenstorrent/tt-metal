# DM perf compare

Measures, on one machine, whether a change moved the speed of a data-movement op and whether the
default call picks the fastest implementation.

- **Q1: did the change alter speed?** Device time of `default` before vs `default` after. Host time
  is reported as detail against the run's measured host resolution and never gets a verdict.
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

Host-side cost (routing, Python entry, program-cache lookup) is invisible to a device-time verdict,
and on shared CI machines host time moves tens of percent between identical runs, so it cannot
stand in. A change that touches the host path needs a paired same-session measurement locally.

## Method

- 5 rounds, after one discarded priming pair. A round runs fresh processes for host time: before
  (default only), after (default only), then after again with `default`, `native`, `codegen` and
  `default_b` (the default measured a second time, reported as `q2_control`). The first two give the
  host change of Q1 and alternate order between rounds; a process that also runs the forced entries
  leaves caches and allocator in another state (measured: about +4% on the same default call).
- Host time, per call and implementation: warm-up, then best of N timed calls, each synced. Profiler
  off. `best_us` is host launch plus device, the time a caller waits.
- Host resolution: a second pair of processes runs the after build against itself on every call; the
  resolution is the largest per-call median change found there, never below 3%. A call whose host
  change exceeds it is listed as `q1_host.moved`; there is no host verdict.
- Device time: separate processes under the device profiler, one per side per round, order
  alternating. Device time is the sum of kernel durations between signposts, best of the reps. The
  before build runs the default only; the after build runs every implementation. The device ops each
  implementation dispatched are recorded, and `default_uses` is whichever of native/codegen has the
  same ops as the default. Both sides run profiler-enabled builds.
- A call's change is the median of its per-round changes (negative is faster).
- Device band: the after build's own device rounds compared with each other (adjacent rounds, absolute
  change, largest per-call median), never below 3%. A band above 5% means the device rounds
  disagreed: the results file marks `noise.reliable: false` and the run should be repeated, since
  "no change" and "default is fastest" from such a run prove nothing.
- Verdict, for Q1 and Q2 alike: `faster` / `slower` only when the median device change clears the
  device band and at least 4 of 5 rounds point the same way; otherwise `no change`.
- Output match: every run hashes its output. Runs being compared must agree, otherwise the call is
  `not comparable` (a run that skips work would look fast).

## Call file

See `calls.py`. One op, a list of calls (`shape`, `dtype`, `layout`, `memory_config`, `args`).

## Report

`python render.py results_*.json > comment.md` renders the PR comment from one results file per op and
arch: what needs a look, the answer to each question, a per-call chart and table, and a collapsed detail
section. The body starts with a marker line so the workflow can update one sticky comment in place. Over
60000 characters it sheds the detail section, then the calls without a finding, then the chart.

## Tests

`pytest tests/ttnn/dm_perf_compare/test_dm_perf_compare.py` covers the comparison logic without a device.
