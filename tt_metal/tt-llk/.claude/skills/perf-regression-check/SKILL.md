---
name: perf-regression-check
description: >
  Check whether the current branch introduced an LLK performance regression versus
  the exact main commit it was branched from (the merge-base, not latest main). Runs
  a chosen perf test on both commits, several iterations each, and reports per-config
  regressions. Use when a developer asks "did my branch make perf worse", "check my
  changes for regressions", or "compare my branch to where I branched off". Needs
  Tenstorrent hardware (runs the real perf sweep).
---

# Perf regression check (branch vs its branch point on main)

Compares **current** = the branch's HEAD against **baseline** = `git merge-base origin/main HEAD`
(the exact main commit the branch was cut from). Runs the same perf test on both, N iterations
each, takes the median per point, and flags points where current is slower by more than the
threshold. Self-contained: uses only `perf_regression_compare.py` (in `.claude/scripts/`), so
it works on any branch whether or not the perf infra is merged.

## Inputs to collect from the user (ask if not given)
- **test** (required): the perf test module, e.g. `perf_math_matmul`. One test per run.
- **arch**: `wormhole` or `blackhole`. Infer from the machine if possible (`tt-smi`), else ask.
- **threshold**: default `0.05` (5%).
- **iterations**: default `3` per side.
- **speed_of_light**: default off. If used, apply the SAME flag to both sides.

## Safety first — never lose the user's work
1. Confirm the working tree is clean: `git status --porcelain`. If dirty, **stop and ask** the
   user to commit or stash; do not proceed with uncommitted changes (the baseline checkout would
   collide). Optionally `git stash push -u` yourself and restore at the end.
2. Record `BRANCH=$(git rev-parse --abbrev-ref HEAD)` and `CURRENT=$(git rev-parse HEAD)`.
3. **Always restore** at the end (and on any failure/interrupt): `git checkout "$BRANCH"` (and
   `git stash pop` if you stashed). Treat this as mandatory cleanup.

## Procedure
Let `WORK=/tmp/perf-regression-check` (mkdir -p) and the compare script be resolved from the repo
root (cwd-independent):
```bash
SCRIPT="$(git rev-parse --show-toplevel)/tt_metal/tt-llk/.claude/scripts/perf_regression_compare.py"
```

**0. Determine commits.**
```bash
git fetch origin main --quiet
BASELINE=$(git merge-base origin/main HEAD)
CURRENT=$(git rev-parse HEAD)
echo "baseline (branch point): $BASELINE"; echo "current: $CURRENT"
```
If `BASELINE == CURRENT`, the branch has no commits vs main — tell the user there's nothing to
compare.

**1. Run the sweep N times on the CURRENT branch.** From the `tests/` directory:
```bash
cd tt_metal/tt-llk/tests
for i in $(seq 1 <iterations>); do
  CHIP_ARCH=<arch> pytest --compile-producer -n 10 -m perf ./python_tests/<test>.py
  CHIP_ARCH=<arch> pytest --compile-consumer -n 15 -m perf ./python_tests/<test>.py
  cp "$(find . -path '*/perf_data/latest/<test>/<test>.csv' | head -1)" "$WORK/current_run_$i.csv"
done
```
Notes: do **not** pass `-x` (it aborts mid-sweep and the combined CSV is partial). The combine
step writes `perf_data/runs/<tag>/<test>/<test>.csv` and points `perf_data/latest` at it;
the `cp` snapshots that run before the next iteration starts its own. If `speed_of_light` is on, add `--speed-of-light` to both pytest calls.

**2. Run the same sweep N times on the BASELINE commit.**
```bash
git checkout "$BASELINE"          # detached HEAD; user's branch is safe
# repeat the same loop, writing "$WORK/baseline_run_$i.csv"
git checkout "$BRANCH"            # restore — do this even if a run failed
```
If the test file does not exist at the baseline (the user *added* it on their branch), there is no
baseline to compare — report that and stop.

**3. Compare + report.**
```bash
python "$SCRIPT" \
  --current  "$WORK/current_run_*.csv" \
  --baseline "$WORK/baseline_run_*.csv" \
  --threshold <threshold> \
  --report "$WORK/regression_report.md" \
  --test <test> --baseline-sha "$BASELINE" --current-sha "$CURRENT"
```
It prints and writes a Markdown report (verdict + top-25 regressions), plus a full
`regression_report.regressions.csv` with every regression. It exits non-zero if any regression is
found.

**4. Present the result.** Show the verdict and the top regressions. Point the user to
`$WORK/regression_report.md` and the companion `.regressions.csv`. If there are many "new points",
explain the branch changed the test's sweep/configs (so those points have no baseline).

## Interpreting results (tell the user)
- Comparison is **per (marker, sweep-config)**; `mean(<run_type>)` cycles, median across iterations.
- A cleanest comparison is when only the **kernel/LLK code** changed, not the test's sweep — then
  every config has a baseline. If the test itself changed, expect "new points".
- **Noise:** 3 iterations + median reduces it, but deltas near the threshold can still be noise;
  suggest more iterations or a higher threshold if results look marginal.
- `TILE_LOOP` marker is the steady-state per-tile cost; `INIT`/`KERNEL` include one-time overheads.

## Failure handling
- Any failure after checkout of the baseline: **restore the branch** before surfacing the error.
- `TENSIX TIMED OUT` / device hang during a run: `tt-smi -r`, then retry that iteration.
- No CSV produced: the test may have been deselected by the marker — check the test carries the
  `perf` marker and the module name is correct.
