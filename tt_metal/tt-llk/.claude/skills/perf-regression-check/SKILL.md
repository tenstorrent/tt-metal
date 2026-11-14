---
name: perf-regression-check
description: >
  Compare one LLK perf test between two commits on this machine — by default the branch
  HEAD against the exact main commit it was branched from (the merge-base, not latest
  main), or any two refs the user names (hash, tag, origin/main, HEAD~5). Runs the sweep
  on both, several iterations each, and reports per-config regressions and improvements.
  Use when a developer asks "did my branch make perf worse", "check my changes for
  regressions", "compare my branch to where I branched off", or "compare perf between
  these two commits", and before you add the llk-perf-gate label or wait for CI to
  check perf. Needs Tenstorrent hardware (runs the real perf sweep).
---

# Perf compare (one test, two commits)

This is the default perf check for a change: run it on the attached card before you
push. Do not add the `llk-perf-gate` label or start a CI perf run to get a verdict: CI
runner time is expensive, and the merge queue runs the gate anyway. Use the CI gate
only when no card is attached, or as the final confirmation.

If the `llk-tools` MCP server is connected (`tools/mcp/README.md`), call its
`perf_compare` tool, and `perf_placement_check` for a regression; they run this same
script. Otherwise run the script yourself:

`perf_compare_commits.sh` does the whole job: resolve both refs, sweep each, compare
median-vs-median, write the report. Do not hand-roll the checkout/sweep loop — the script
already handles worktrees, caching, interleaving and cleanup.

```bash
SCRIPT="$(git rev-parse --show-toplevel)/tt_metal/tt-llk/.claude/scripts/perf_compare_commits.sh"

# branch HEAD vs the commit the branch was cut from (the default)
"$SCRIPT" <arch> <test>

# any two commits
"$SCRIPT" <arch> <test> --baseline <ref> --current <ref>

# a stage-2 code position: the TRISC code moved by 4, 24, 44 or 60 bytes (needs #62981)
"$SCRIPT" <arch> <test> --code-shift 24
```

It prints the report, writes it under `~/.cache/tt-llk-perf-compare/reports/…`, and exits
non-zero when a regression is found. `--dry-run` shows what it would measure (and what is
already cached) without running a sweep — cheap, use it to confirm the refs first.

## Inputs to collect from the user (ask only what you cannot infer)
- **test** (required): the perf test module, e.g. `perf_math_matmul`. One test per run.
- **arch** (required): `wormhole` or `blackhole`. Infer from the machine (`tt-smi`) if possible.
- **baseline / current**: any refs. Default: baseline = `git merge-base origin/main HEAD`,
  current = `HEAD`. If the user names commits ("compare abc123 and def456"), pass both.
- **threshold**: `--threshold`, a fraction. Default: the merge gate's, `0.02` on
  blackhole and `0.13` on wormhole. `--module-threshold` sets it per module; the
  default on wormhole is `perf_matmul=0.25,perf_math_matmul=0.25`.
- **run types**: `--run-types`, default `L1_TO_L1` (as the gates). `all` measures and
  compares every run type.
- **min cycles**: `--min-cycles`, default `30`. A point is a regression only when it is
  **both** more than `--threshold` slower **and** more than `--min-cycles` slower. The
  percentage clause alone fires on small markers such as `INIT`; the cycle clause alone
  fires on large `TILE_LOOP` values. `0` disables the cycle clause.
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
- **As CI:** `TT_LLK_DISABLE_ASSERTS=1`, `LLK_PERF_RUN_TYPES` set to the run types, and
  `--dist loadgroup` for the run step. The cache key holds the run types, `CONSUMER_JOBS`,
  the asserts setting and the code shift.
- **One sweep per card.** A sweep holds `/tmp/tt-llk-test-<arch>.lock`, the lock that
  `run_test.sh` takes.

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
  a 3-iteration run only measures 2 more per side).
- **Wormhole on main is noisy.** Results move from run to run with test order and worker
  assignment until #60892, #58068 and #60894 are merged. Before you report a Wormhole
  regression, repeat with more iterations, or with `CONSUMER_JOBS=1` (one fixed order),
  and run the placement check. A move that is gone in a repeat is noise.
- **Placement:** a change can move code, and the same loop then runs faster or slower at
  its new address. Stage 2 of the gate (#62981, #63686) re-measures each regression with
  the code at 0, 4, 24, 44 and 60 bytes: slower at every position is a regression, at
  some positions only is placement. `perf_placement_check` (MCP) or `--code-shift` (script)
  does the same here. Both commits need the `LLK_CODE_SHIFT` knob (#62981).
- **Local and gate baselines differ.** Here the baseline is a commit measured on this
  card. The gates use the warehouse baseline (pinned, or `merge_baseline` in the merge
  queue), so an old warehouse baseline can add moves that the local compare does not show.
- The further apart the two commits are, the more unrelated change is in the delta. For
  "did *this* work regress perf", the merge-base default is the honest baseline.
- `TILE_LOOP` is the steady-state per-tile cost; `INIT`/`KERNEL` include one-time overheads.
- The schema version is `PERF_TEST_SCHEMAS[...]["version"]` in
  `helpers/perf/test_schemas.py`, not a CSV column. It changes when columns
  change. A schema version change, or a `tile_cnt` change that starts counting
  both operand tiles, is an accounting change, not a kernel regression. A
  `tile_cnt` accounting change can make rows incomparable without a version
  bump. Compare cycle numbers only inside one schema version. A ~2× drop on
  MATH_ISOLATE after that accounting change is about half a result tile, not
  a speedup.
  At 16-bit, `dest_acc=Yes` can also double dest handshakes.

## Failure handling
- `TENSIX TIMED OUT` / device hang: `tt-smi -r`, then re-run the same command — completed
  iterations are cached, so it resumes instead of starting over.
- Missing header at compile time on an older commit: that commit's layout needs more than
  the default sparse checkout — re-run with `SPARSE_PATHS=` (empty) for a full checkout.
- "no perf CSV": the test may have been deselected — check the module name and that the
  test carries the `perf` marker.
- "does not exist at <sha>": the test was added after that commit; pick a baseline that has it.
