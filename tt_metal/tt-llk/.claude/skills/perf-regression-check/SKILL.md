---
name: perf-regression-check
description: >
  Compare one LLK perf test between two commits on this machine — by default the branch
  HEAD against the exact main commit it was branched from (the merge-base, not latest
  main), or any two refs the user names (hash, tag, origin/main, HEAD~5). Runs the sweep
  on both, several iterations each, and reports per-config regressions and improvements.
  Use when a developer asks "did my branch make perf worse", "check my changes for
  regressions", "compare my branch to where I branched off", or "compare perf between
  these two commits". Needs Tenstorrent hardware (runs the real perf sweep).
---

# Perf compare (one test, two commits)

`perf_compare_commits.sh` does the whole job: resolve both refs, sweep each, compare
median-vs-median, write the report. Do not hand-roll the checkout/sweep loop — the script
already handles worktrees, caching, interleaving and cleanup.

```bash
SCRIPT="$(git rev-parse --show-toplevel)/tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh"

# branch HEAD vs the commit the branch was cut from (the default)
"$SCRIPT" <arch> <test>

# any two commits
"$SCRIPT" <arch> <test> --baseline <ref> --current <ref>
```

It prints the report, writes it under `~/.cache/tt-llk-perf-compare/reports/…`, and exits
non-zero when a regression is found. `--dry-run` shows what it would measure (and what is
already cached) without running a sweep — cheap, use it to confirm the refs first.

## Inputs to collect from the user (ask only what you cannot infer)
- **test** (required): the perf test module, e.g. `perf_math_matmul`. One test per run.
- **arch** (required): `wormhole` or `blackhole`. Infer from the machine (`tt-smi`) if possible.
- **baseline / current**: any refs. Default: baseline = `git merge-base origin/main HEAD`,
  current = `HEAD`. If the user names commits ("compare abc123 and def456"), pass both.
- **threshold**: `--threshold`, default `0.05` (5%).
- **iterations**: `--iterations`, default `3` per side.
- **speed of light**: `--speed-of-light`. The script applies it to both sides.

## What the script guarantees (so you do not have to)
- **Your checkout is never touched.** Each commit is a separate sparse git worktree with
  its own build tree — no branch switch, no stash, a dirty tree is fine, and an interrupted
  run cannot leave the user on a detached HEAD. Worktrees are removed on exit.
- **Only committed code is measured.** Uncommitted edits are invisible to a ref; the script
  says so. If the user wants their working tree measured, they must commit it first.
- **Runs are cached** per (arch, test, variant, commit), so a commit already measured on this
  machine costs nothing to reuse — comparing A-vs-B then B-vs-C only sweeps C. `--refresh`
  re-measures. Cached runs from another host are refused (cycles are not comparable).
- **Iterations are interleaved** baseline, current, baseline, current…, so machine drift
  hits both sides equally instead of biasing one.

## Present the result
Show the verdict, the top regressions, and the improvements if any. Point at
`regression_report.md` plus its companions `.regressions.csv` (every regression) and
`.points.csv` (every compared point, full config).

## Interpreting results (tell the user)
- Comparison is **per (marker, sweep-config)** on `mean(<run_type>)` cycles, median across
  iterations. A point exists on both sides or it is a "new point" (no baseline).
- Cleanest when only **kernel/LLK code** differs between the commits. If the test's sweep
  changed, expect "new points" — those are reported, never counted as regressions.
- **Noise:** 3 iterations + median reduces it, but deltas near the threshold can still be
  noise. Suggest more iterations (cheap — cached runs are reused, so `--iterations 5` after
  a 3-iteration run only measures 2 more per side) or a higher threshold.
- The further apart the two commits are, the more unrelated change is in the delta. For
  "did *this* work regress perf", the merge-base default is the honest baseline.
- `TILE_LOOP` is the steady-state per-tile cost; `INIT`/`KERNEL` include one-time overheads.

## Failure handling
- `TENSIX TIMED OUT` / device hang: `tt-smi -r`, then re-run the same command — completed
  iterations are cached, so it resumes instead of starting over.
- Missing header at compile time on an older commit: that commit's layout needs more than
  the default sparse checkout — re-run with `SPARSE_PATHS=` (empty) for a full checkout.
- "no perf CSV": the test may have been deselected — check the module name and that the
  test carries the `perf` marker.
- "does not exist at <sha>": the test was added after that commit; pick a baseline that has it.
