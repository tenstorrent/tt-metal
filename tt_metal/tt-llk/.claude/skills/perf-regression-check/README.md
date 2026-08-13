# perf-regression-check

Locally check whether **your branch introduced an LLK performance regression** versus the exact
`main` commit you branched from — before you push.

It runs a perf test on **your HEAD** and on **`git merge-base origin/main HEAD`** (the commit you
branched off, *not* latest main), a few iterations each, and reports per-config slowdowns.

## Requirements
- A Tenstorrent machine — it runs the real perf sweep on hardware.
- A **clean working tree** — commit or stash first. The tool checks out the baseline commit and
  restores your branch afterward, so uncommitted work would get in the way.
- Your usual LLK Python test environment active.

## Use it (in Claude Code)
Either run the slash command:
```
/perf-regression-check perf_math_matmul
```
…or, if it isn't listed (these skills live in a nested folder), just say:
> Read `tt_metal/tt-llk/.claude/skills/perf-regression-check/SKILL.md` and follow it for
> `perf_math_matmul` (arch: blackhole).

Claude figures out the two commits, runs the sweep on both, compares, and writes the report.

## Inputs
| input | required | default |
|---|---|---|
| **test** | yes | — e.g. `perf_math_matmul` (one test) |
| **arch** | inferred, else asked | `wormhole` or `blackhole` |
| **threshold** | no | `0.05` (5%) |
| **iterations** | no | `3` per side (median-vs-median) |

## Output
- `/tmp/perf-regression-check/regression_report.md` — verdict + the top regressions.
- `…/regression_report.regressions.csv` — every regression, full config.
- Exit code is non-zero when a regression is found (usable in scripts).

## Reading the report
- Comparison is per **(marker, sweep-config)**, on `mean(<run_type>)` cycles, **median across
  iterations**.
- Cleanest when only your **kernel / LLK code** changed (not the test's sweep) — then every config
  has a baseline. If you changed the test itself, expect "new points" (no baseline; not counted as
  regressions).
- **Noise:** deltas right at the threshold can be measurement noise. Raise the iterations or the
  threshold if a result looks marginal.

## Cost
It runs the test's full sweep on **both** commits × iterations, and rebuilds at the baseline — for
a test like `perf_math_matmul` expect **~15 min**. Pick a focused test; that's why a test name is
required.

## How it works
`sweep(HEAD)` + `sweep(merge-base(origin/main, HEAD))` → median-vs-median compare via the
self-contained `perf_regression_compare.py`, which reads the raw `perf_data` CSVs directly (no
Parquet, no database — works on any branch, merged or not).
