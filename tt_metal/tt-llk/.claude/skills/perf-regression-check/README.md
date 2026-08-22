# perf-regression-check

Compare **one LLK perf test between two commits** on your machine, before you push.

By default it compares your **HEAD** against **`git merge-base origin/main HEAD`** — the commit
you branched off, *not* latest main. Name any two refs and it compares those instead.

## Requirements
- A Tenstorrent machine — it runs the real perf sweep on hardware.
- Your usual LLK Python test environment active.
- A clean working tree is **not** required: each commit is measured in its own git worktree, so
  your checkout is never touched. Note the flip side — uncommitted edits are not measured.

## Use it
```bash
S=tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh

$S blackhole perf_math_matmul                                   # branch vs branch point
$S blackhole perf_math_matmul --baseline v0.60.0                # vs a tag
$S wormhole  perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6   # two hashes
$S blackhole perf_math_matmul --baseline HEAD~5 --dry-run       # what would it measure?
```

In Claude Code, `/perf-regression-check perf_math_matmul` (or "compare perf of
`perf_math_matmul` between abc123 and def456") drives the same script.

## Options
| option | default | meaning |
|---|---|---|
| `--baseline <ref>` | `merge-base origin/main HEAD` | what to compare against |
| `--current <ref>` | `HEAD` | what to judge |
| `--iterations <N>` | `3` | sweeps per side; median-vs-median |
| `--threshold <T>` | `0.05` | 5% — flagged in both directions |
| `--speed-of-light` | off | compile-time-parameter build, applied to both sides |
| `--refresh` | off | ignore cached runs and re-measure |
| `--dry-run` | off | resolve refs, report the plan, measure nothing |
| `--out-dir <dir>` | under `$PERF_COMPARE_HOME` | where the report goes |

Env knobs: `PERF_COMPARE_HOME` (cache + reports, default `~/.cache/tt-llk-perf-compare`),
`WORK_ROOT` (worktrees and build trees, default `/tmp/tt-llk-perf-compare`), `PRODUCER_JOBS`,
`CONSUMER_JOBS`, `RESET_BETWEEN_RUNS=1`, `SPARSE_PATHS`, `ALLOW_CROSS_HOST=1`.

## Output
- `regression_report.md` — verdict, top regressions, top improvements, new points.
- `regression_report.regressions.csv` — every regression, full config.
- `regression_report.points.csv` — every compared point with its delta.
- Exit code is non-zero when a regression is found (usable in scripts).

## Reading the report
- Comparison is per **(marker, sweep-config)**, on `mean(<run_type>)` cycles, **median across
  iterations**. `Δ` is signed: `+` slower on current, `−` faster.
- Cleanest when only **kernel / LLK code** differs between the commits — then every config has
  a baseline. If the test's sweep changed, expect "new points" (no baseline; never counted as
  regressions).
- **Noise:** deltas right at the threshold can be measurement noise. Raise `--iterations` — a
  re-run reuses the sweeps you already paid for and only measures the new ones.

## Cost
A full sweep per side per iteration, plus a build per commit — for a test like
`perf_math_matmul` expect **~15 min** the first time. After that, any comparison involving an
already-measured commit reuses its cached runs, so a chain of comparisons (A-vs-B, then
B-vs-C) only pays for the new commit.

## How it works
For each side: a sparse `git worktree` at that commit (tt-llk plus the trees a kernel build
includes from outside it), a private build tree, then N producer/consumer sweeps — interleaved
between the two sides so thermals and other tenants bias neither. The two sets of CSVs go to
`perf_regression_compare.py`, which reads the raw `perf_data` CSVs directly (no Parquet, no
database — works on any commit, merged or not) and reports median-vs-median per point.
