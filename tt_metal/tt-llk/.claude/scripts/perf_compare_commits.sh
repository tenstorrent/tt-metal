#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Compare ONE LLK perf test between TWO commits, on this machine, right now.
#
# Usage:
#   perf_compare_commits.sh <arch> <test> [options]
#
#   --baseline <ref>     what to compare against  (default: merge-base origin/main HEAD)
#   --current  <ref>     what to judge            (default: HEAD)
#   --iterations <N>     runs per side            (default: 3, median-vs-median)
#   --threshold <T>      relative slowdown that counts  (default: 0.02 = 2%)
#   --min-cycles <N>     absolute slowdown it must ALSO exceed, in cycles
#                        (default: 30; 0 disables the clause)
#   --speed-of-light     compile-time-parameter build; applied to BOTH sides
#   --refresh            ignore cached runs and re-measure both sides
#   --keep-worktrees     leave the per-commit worktrees behind (for debugging)
#   --out-dir <dir>      where the report goes
#   --dry-run            resolve the refs, say what would be measured, measure nothing
#
# Examples:
#   perf_compare_commits.sh blackhole perf_math_matmul                      # branch vs branch point
#   perf_compare_commits.sh blackhole perf_math_matmul --baseline v0.60.0   # any two refs
#   perf_compare_commits.sh wormhole perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6
#
# Any ref git understands works on either side: a hash, a tag, origin/main, HEAD~5.
#
# Your checkout is never touched. Each commit is materialized as its own sparse
# git worktree (see SPARSE_PATHS) with its own build tree, so no branch is checked
# out, a dirty working tree is fine, and a crash cannot leave you on a detached
# HEAD. What is measured is the COMMITTED state of each ref -- your uncommitted
# edits are not part of the comparison.
#
# Runs are cached per (arch, test, variant, commit) under $PERF_COMPARE_HOME, so
# comparing A-vs-B and then B-vs-C only measures C. Iterations of the two sides
# are interleaved (baseline, current, baseline, current, ...) so machine drift --
# thermals, other tenants -- hits both sides equally instead of biasing one.
#
# Environment:
#   PERF_COMPARE_HOME=<dir>   cache + reports (default: ~/.cache/tt-llk-perf-compare)
#   WORK_ROOT=<dir>           worktrees + build trees, large and disposable
#                             (default: /tmp/tt-llk-perf-compare)
#   PRODUCER_JOBS / CONSUMER_JOBS   pytest -n for each phase (default 10 / 15).
#                             Keep these constant across the commits you compare:
#                             they are not part of the cache key, so changing them
#                             mid-comparison silently compares unlike runs.
#   RESET_BETWEEN_RUNS=1      tt-smi -r between iterations
#   ALLOW_CROSS_HOST=1        reuse cached runs measured on another host (invalid
#                             comparison; only for inspecting old data)
#   SPARSE_PATHS=<paths>      what each worktree checks out. The default is tt-llk plus
#                             the trees a kernel build includes from outside it
#                             (tt_metal/hw, tt_metal/hostdevcommon, the ttnn
#                             experimental kernels) -- about a fifth of the repo. If a
#                             commit needs more and a compile fails on a missing
#                             header, add the path here, or set SPARSE_PATHS= (empty)
#                             for a full checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_PY="$SCRIPT_DIR/perf_regression_compare.py"
LLK_PRIMARY="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(git -C "$LLK_PRIMARY" rev-parse --show-toplevel)"
# tt-llk relative to the repo root -- the same path inside every worktree.
LLK_RELPATH="${LLK_PRIMARY#"$REPO_ROOT"/}"

BASE_HOME="${PERF_COMPARE_HOME:-$HOME/.cache/tt-llk-perf-compare}"
WORK_ROOT="${WORK_ROOT:-/tmp/tt-llk-perf-compare}"
PRODUCER_JOBS="${PRODUCER_JOBS:-10}"
CONSUMER_JOBS="${CONSUMER_JOBS:-15}"
# Kernel compiles reach out of tt-llk: ../../hw/inc, ../../hw/ckernels/<arch>/metal,
# ../../hostdevcommon/api and ../../../ttnn/cpp/ttnn/operations/experimental (see
# TestConfig.INCLUDES). A worktree missing those builds nothing.
SPARSE_PATHS="${SPARSE_PATHS-$LLK_RELPATH tt_metal/hw tt_metal/hostdevcommon ttnn/cpp/ttnn/operations/experimental}"

[ "$LLK_RELPATH" != "$LLK_PRIMARY" ] || {
    echo "error: $LLK_PRIMARY is not under the repo root $REPO_ROOT" >&2
    exit 2
}

die() {
    echo "error: $*" >&2
    exit 2
}

usage() {
    cat >&2 <<'EOF'
Usage: perf_compare_commits.sh <arch> <test> [options]

  --baseline <ref>     what to compare against  (default: merge-base origin/main HEAD)
  --current  <ref>     what to judge            (default: HEAD)
  --iterations <N>     runs per side            (default: 3, median-vs-median)
  --threshold <T>      relative slowdown that counts  (default: 0.02 = 2%)
  --min-cycles <N>     absolute slowdown it must ALSO exceed, in cycles (default: 30)
  --speed-of-light     compile-time-parameter build; applied to BOTH sides
  --refresh            ignore cached runs and re-measure both sides
  --keep-worktrees     leave the per-commit worktrees behind (for debugging)
  --out-dir <dir>      where the report goes
  --dry-run            resolve the refs, say what would be measured, measure nothing

  perf_compare_commits.sh blackhole perf_math_matmul                     # branch vs branch point
  perf_compare_commits.sh blackhole perf_math_matmul --baseline v0.60.0  # any two refs
  perf_compare_commits.sh wormhole perf_math_matmul --baseline 1a2b3c4 --current 9f8e7d6
EOF
    exit 2
}

# --- arguments
[ $# -ge 2 ] || usage
ARCH="$1"
TEST="$2"
shift 2

BASELINE_REF=""
CURRENT_REF="HEAD"
ITERATIONS=3
# Measured on a 5-run baseline of unchanged code, not guessed. See
# docs/perf_evaluation/results/blackhole-nonsol/README.md and the constants in
# perf_regression_compare.py.
THRESHOLD=0.02
MIN_CYCLES=30
SPEED_OF_LIGHT=0
REFRESH=0
KEEP_WORKTREES=0
DRY_RUN=0
OUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
    --baseline)
        BASELINE_REF="${2:?--baseline needs a ref}"
        shift 2
        ;;
    --current)
        CURRENT_REF="${2:?--current needs a ref}"
        shift 2
        ;;
    --iterations)
        ITERATIONS="${2:?--iterations needs a number}"
        shift 2
        ;;
    --threshold)
        THRESHOLD="${2:?--threshold needs a number}"
        shift 2
        ;;
    --min-cycles)
        MIN_CYCLES="${2:?--min-cycles needs a number}"
        shift 2
        ;;
    --out-dir)
        OUT_DIR="${2:?--out-dir needs a path}"
        shift 2
        ;;
    --speed-of-light)
        SPEED_OF_LIGHT=1
        shift
        ;;
    --refresh)
        REFRESH=1
        shift
        ;;
    --keep-worktrees)
        KEEP_WORKTREES=1
        shift
        ;;
    --dry-run)
        DRY_RUN=1
        shift
        ;;
    -h | --help) usage ;;
    *) die "unknown option '$1'" ;;
    esac
done

case "$ARCH" in
wormhole | blackhole) ;;
*) die "arch must be 'wormhole' or 'blackhole', got '$ARCH'" ;;
esac

[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || die "--iterations must be a positive integer"

TEST="${TEST%.py}"
TEST="$(basename "$TEST")"

VARIANT=runtime
PYTEST_VARIANT_ARGS=()
if [ "$SPEED_OF_LIGHT" = 1 ]; then
    VARIANT=sol
    PYTEST_VARIANT_ARGS+=(--speed-of-light)
fi

# --- resolve both sides to full commit hashes
cd "$LLK_PRIMARY"
if [ -z "$BASELINE_REF" ]; then
    git fetch origin main --quiet || echo "warning: could not fetch origin/main; using the local ref" >&2
    BASELINE_REF="$(git merge-base origin/main HEAD)"
    BASELINE_LABEL="branch point on main"
else
    BASELINE_LABEL="$BASELINE_REF"
fi
CURRENT_LABEL="$CURRENT_REF"

resolve() {
    git rev-parse --verify --quiet "$1^{commit}" || die "not a commit this repo knows: '$1'"
}
BASELINE_SHA="$(resolve "$BASELINE_REF")"
CURRENT_SHA="$(resolve "$CURRENT_REF")"
BASELINE_SHORT="${BASELINE_SHA:0:12}"
CURRENT_SHORT="${CURRENT_SHA:0:12}"

[ "$BASELINE_SHA" != "$CURRENT_SHA" ] ||
    die "both sides resolve to $BASELINE_SHORT -- nothing to compare"

# A ref only ever names a commit, so uncommitted work is invisible here. Say so
# rather than let someone read the verdict as covering their working tree. .claude/
# is agent tooling (this script included) and never a build input, so changes there
# are not worth a warning.
if [ -n "$(git status --porcelain -- "$LLK_PRIMARY" ':(exclude,glob)**/.claude/**')" ]; then
    echo "note: uncommitted changes under $LLK_RELPATH are NOT measured -- each side is"
    echo "      the committed state of its ref. Commit them if they should count."
fi

REPORT_DIR="${OUT_DIR:-$BASE_HOME/reports/$ARCH/$TEST/${BASELINE_SHORT}_vs_${CURRENT_SHORT}_$VARIANT}"
mkdir -p "$REPORT_DIR" "$WORK_ROOT"

# Everything that decides the verdict, stated before any of it runs. A reader must
# not have to guess which defaults applied, which refs a shorthand resolved to, or
# what the pass/fail rule actually is.
cat <<BANNER

  Perf compare
  ------------------------------------------------------------------
  test          $TEST
  arch          $ARCH
  build         $VARIANT $([ "$SPEED_OF_LIGHT" = 1 ] && echo "(--speed-of-light, applied to BOTH sides)" || echo "(runtime parameters; pass --speed-of-light for the compile-time build)")
  iterations    $ITERATIONS per side, interleaved; median vs median

  baseline      $BASELINE_SHORT  $BASELINE_LABEL
  current       $CURRENT_SHORT  $CURRENT_LABEL

  verdict rule  slower by more than $(awk -v t="$THRESHOLD" 'BEGIN{printf "%g", t*100}')% AND more than $MIN_CYCLES cycles
                Both must hold. The percentage alone fires on small markers such
                as INIT, where a few cycles of jitter is a large percentage; the
                cycle count alone fires on large markers, where a real 2% is
                thousands of cycles. Defaults are measured, not guessed --
                docs/perf_evaluation/results/blackhole-nonsol/README.md.

  compared per  (marker, run type, sweep config) -- every row, matched
  measured on   $(hostname)
  parallelism   producer -n $PRODUCER_JOBS, consumer -n $CONSUMER_JOBS
  cache         $BASE_HOME
  report        $REPORT_DIR
  ------------------------------------------------------------------

BANNER

# --- worktrees: one per commit, removed on exit unless asked to keep them
WORKTREES=()
cleanup() {
    local wt
    if [ "$KEEP_WORKTREES" = 1 ]; then
        if [ ${#WORKTREES[@]} -gt 0 ]; then
            echo "note: worktrees kept: ${WORKTREES[*]}"
        fi
        return 0
    fi
    for wt in ${WORKTREES+"${WORKTREES[@]}"}; do
        git -C "$REPO_ROOT" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    done
    return 0
}
trap cleanup EXIT INT TERM

make_worktree() {
    # Worktree at commit $1 (sparse unless SPARSE_PATHS is empty); prints its path.
    # Nothing here checks out a branch, so the user's working tree is never involved.
    local sha="$1"
    local dir="$WORK_ROOT/wt-${sha:0:12}"
    if [ -d "$dir/$LLK_RELPATH/tests/python_tests" ]; then # resume
        echo "$dir"
        return
    fi
    rm -rf "$dir"
    # A run killed outright (no trap) leaves the path registered but gone; without
    # this, 'worktree add' refuses the same path forever.
    git -C "$REPO_ROOT" worktree prune
    if [ -n "$SPARSE_PATHS" ]; then
        git -C "$REPO_ROOT" worktree add --no-checkout --detach "$dir" "$sha" >/dev/null
        # sparse-checkout only sets the patterns; the checkout after it populates.
        if ! (git -C "$dir" sparse-checkout set --cone $SPARSE_PATHS &&
            git -C "$dir" checkout --detach "$sha") >/dev/null 2>&1; then
            # Old git, or a pattern it dislikes: fall back to a full checkout.
            git -C "$REPO_ROOT" worktree remove --force "$dir" 2>/dev/null || rm -rf "$dir"
            git -C "$REPO_ROOT" worktree add --detach "$dir" "$sha" >/dev/null
        fi
    else
        git -C "$REPO_ROOT" worktree add --detach "$dir" "$sha" >/dev/null
    fi
    [ -d "$dir/$LLK_RELPATH/tests/python_tests" ] ||
        die "worktree for ${sha:0:12} has no $LLK_RELPATH/tests/python_tests"
    # A path that existed when this script was written may not exist at an old
    # commit; say so once here rather than as a puzzling missing-header error.
    local p
    for p in $SPARSE_PATHS; do
        if [ ! -e "$dir/$p" ]; then
            echo "warning: '$p' does not exist at ${sha:0:12}; if a compile fails on a" >&2
            echo "         missing header, set SPARSE_PATHS for that commit's layout" >&2
        fi
    done
    echo "$dir"
}

ensure_sfpi() {
    # tests/sfpi is downloaded, not tracked, so a fresh worktree has none. Reuse
    # the primary checkout's toolchain when it is the version this commit pins,
    # otherwise let the commit's own setup script fetch the right one.
    local llk="$1" sha="$2"
    local want have="" primary_ver=""
    want="$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" "$llk/tests/sfpi-version" | head -1)"
    if [ -f "$llk/tests/sfpi/sfpi.version" ]; then
        have="$(cat "$llk/tests/sfpi/sfpi.version")"
    fi
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
        return 0
    fi

    if [ -f "$LLK_PRIMARY/tests/sfpi/sfpi.version" ]; then
        primary_ver="$(cat "$LLK_PRIMARY/tests/sfpi/sfpi.version")"
    fi
    if [ -n "$want" ] && [ "$primary_ver" = "$want" ]; then
        rm -rf "$llk/tests/sfpi"
        ln -sfn "$LLK_PRIMARY/tests/sfpi" "$llk/tests/sfpi"
        echo "   sfpi $want: reusing the toolchain from your checkout"
        return 0
    fi
    echo "   sfpi ${want:-?} for ${sha:0:12} differs from your checkout's (${primary_ver:-none}); fetching"
    (cd "$llk" && ./tests/setup_testing_env.sh) ||
        die "could not install the sfpi toolchain ${want:-?} that ${sha:0:12} pins"
}

# --- cache: runs are identified by what can change a measured cycle
cache_dir_for() {
    echo "$BASE_HOME/runs/$ARCH/$TEST/$VARIANT/$1"
}

check_cache_provenance() {
    # Cycles from another machine are not comparable to cycles from this one, and
    # sweep parallelism is not part of the cache key, so check both here.
    local meta="$1/meta.txt" cached_host cached_jobs
    if [ ! -f "$meta" ]; then
        return 0
    fi
    cached_jobs="$(sed -n 's/^consumer_jobs: //p' "$meta" | head -1)"
    if [ -n "$cached_jobs" ] && [ "$cached_jobs" != "$CONSUMER_JOBS" ]; then
        echo "warning: cached runs in $1 used CONSUMER_JOBS=$cached_jobs, this run uses" >&2
        echo "         $CONSUMER_JOBS -- re-run with --refresh to compare like with like" >&2
    fi
    cached_host="$(sed -n 's/^host: //p' "$meta" | head -1)"
    if [ -n "$cached_host" ] && [ "$cached_host" != "$(hostname)" ]; then
        if [ -z "${ALLOW_CROSS_HOST:-}" ]; then
            die "cached runs in $1 were measured on '$cached_host', not '$(hostname)'.
       Cross-machine cycles are not comparable. Re-measure with --refresh, or set
       ALLOW_CROSS_HOST=1 if you only want to look at the old numbers."
        fi
        echo "warning: reusing runs measured on '$cached_host' -- the comparison is not valid" >&2
    fi
}

write_meta() {
    {
        echo "commit: $2"
        echo "arch: $ARCH"
        echo "test: $TEST"
        echo "variant: $VARIANT"
        echo "host: $(hostname)"
        echo "producer_jobs: $PRODUCER_JOBS"
        echo "consumer_jobs: $CONSUMER_JOBS"
    } >"$1/meta.txt"
}

measure_iteration() {
    # One producer+consumer sweep of $TEST at commit $1, snapshotted to $4.
    local sha="$1" wt="$2" side="$3" out_csv="$4"
    local llk="$wt/$LLK_RELPATH"
    local perf_data="$llk/perf_data"
    local build_root="$WORK_ROOT/build-${sha:0:12}"
    local test_file="$llk/tests/python_tests/$TEST.py"

    if [ ! -f "$test_file" ]; then
        die "$TEST.py does not exist at ${sha:0:12} (${side}). If the test was ADDED after
       that commit there is nothing to compare against; pick a ref that has it."
    fi
    mkdir -p "$build_root"

    # A stale perf_data would let the previous iteration's CSV stand in for a
    # sweep that failed to report this time.
    rm -rf "$perf_data"
    (
        cd "$llk/tests/python_tests"
        export LLK_HOME="$llk"
        export RUNNER_TEMP="$build_root" # -> ARTEFACTS_DIR, private per commit
        export CHIP_ARCH="$ARCH"
        # No -x: it aborts mid-sweep and the combined CSV is silently partial.
        pytest -q --override-ini=log_cli=false --compile-producer \
            -n "$PRODUCER_JOBS" -m perf \
            ${PYTEST_VARIANT_ARGS+"${PYTEST_VARIANT_ARGS[@]}"} "./$TEST.py"
        pytest -q --override-ini=log_cli=false --compile-consumer \
            -n "$CONSUMER_JOBS" -m perf \
            ${PYTEST_VARIANT_ARGS+"${PYTEST_VARIANT_ARGS[@]}"} "./$TEST.py"
    )

    # Where a sweep lands depends on the commit being measured, and a compare
    # normally straddles the change: newer trees write perf_data/runs/<tag>/ and
    # point perf_data/latest at it, older ones write perf_data/<test>/ directly.
    local csv="$perf_data/latest/$TEST/$TEST.csv"
    if [ ! -f "$csv" ]; then
        csv="$perf_data/$TEST/$TEST.csv"
    fi
    if [ ! -f "$csv" ]; then
        csv="$(find "$llk" -path "*/perf_data/*$TEST/$TEST.csv" | head -1)"
    fi
    if [ -z "$csv" ] || [ ! -f "$csv" ]; then
        die "no perf CSV from ${sha:0:12} (${side}). Was every case deselected -- does
       $TEST carry the 'perf' marker?"
    fi
    cp "$csv" "$out_csv"
}

# --- plan: which iterations are missing per side (cache hit = nothing to run)
BASELINE_CACHE="$(cache_dir_for "$BASELINE_SHA")"
CURRENT_CACHE="$(cache_dir_for "$CURRENT_SHA")"
if [ "$REFRESH" = 1 ]; then
    rm -rf "$BASELINE_CACHE" "$CURRENT_CACHE"
fi
mkdir -p "$BASELINE_CACHE" "$CURRENT_CACHE"
check_cache_provenance "$BASELINE_CACHE"
check_cache_provenance "$CURRENT_CACHE"

missing_for() {
    local dir="$1" i
    for i in $(seq 1 "$ITERATIONS"); do
        [ -f "$dir/run_$i.csv" ] || echo "$i"
    done
}
BASELINE_MISSING=($(missing_for "$BASELINE_CACHE"))
CURRENT_MISSING=($(missing_for "$CURRENT_CACHE"))

cached_count() { find "$1" -name 'run_*.csv' | wc -l | tr -d ' '; }
echo "   cached: baseline $(cached_count "$BASELINE_CACHE") run(s), current $(cached_count "$CURRENT_CACHE") run(s)"
echo "   to run: baseline ${#BASELINE_MISSING[@]} sweep(s), current ${#CURRENT_MISSING[@]} sweep(s)"

if [ "$DRY_RUN" = 1 ]; then
    echo "== dry run, stopping here"
    exit 0
fi

if [ ${#BASELINE_MISSING[@]} -gt 0 ]; then
    echo "== worktree for baseline $BASELINE_SHORT"
    BASELINE_WT="$(make_worktree "$BASELINE_SHA")"
    WORKTREES+=("$BASELINE_WT")
    ensure_sfpi "$BASELINE_WT/$LLK_RELPATH" "$BASELINE_SHA"
    write_meta "$BASELINE_CACHE" "$BASELINE_SHA"
fi
if [ ${#CURRENT_MISSING[@]} -gt 0 ]; then
    echo "== worktree for current $CURRENT_SHORT"
    CURRENT_WT="$(make_worktree "$CURRENT_SHA")"
    WORKTREES+=("$CURRENT_WT")
    ensure_sfpi "$CURRENT_WT/$LLK_RELPATH" "$CURRENT_SHA"
    write_meta "$CURRENT_CACHE" "$CURRENT_SHA"
fi

# --- measure, interleaved so drift is shared between the two sides
TOTAL=$((${#BASELINE_MISSING[@]} + ${#CURRENT_MISSING[@]}))
DONE=0
for i in $(seq 1 "$ITERATIONS"); do
    for side in baseline current; do
        if [ "$side" = baseline ]; then
            if [ -f "$BASELINE_CACHE/run_$i.csv" ]; then
                continue
            fi
            sha="$BASELINE_SHA" wt="$BASELINE_WT" cache="$BASELINE_CACHE"
        else
            if [ -f "$CURRENT_CACHE/run_$i.csv" ]; then
                continue
            fi
            sha="$CURRENT_SHA" wt="$CURRENT_WT" cache="$CURRENT_CACHE"
        fi
        DONE=$((DONE + 1))
        echo "== [$DONE/$TOTAL] $side ${sha:0:12}, iteration $i ($(date -u +%H:%M:%SZ))"
        measure_iteration "$sha" "$wt" "$side" "$cache/run_$i.csv"
        if [ -n "${RESET_BETWEEN_RUNS:-}" ] && [ "$DONE" -lt "$TOTAL" ]; then
            tt-smi -r
        fi
    done
done

# --- compare. Every cached run counts, not just this invocation's iterations:
# more samples per side is a better median, and the two sides may differ in count.
echo "== comparing"
set +e
python3 "$COMPARE_PY" \
    --baseline "$BASELINE_CACHE/run_*.csv" \
    --current "$CURRENT_CACHE/run_*.csv" \
    --threshold "$THRESHOLD" \
    --min-cycles "$MIN_CYCLES" \
    --report "$REPORT_DIR/regression_report.md" \
    --test "$TEST" \
    --baseline-sha "$BASELINE_SHA" --current-sha "$CURRENT_SHA" \
    --baseline-label "$BASELINE_LABEL" --current-label "$CURRENT_LABEL"
STATUS=$?
set -e

# The compare exits 1 for "regressions found", which a crash would look like too.
# The report only exists if it got far enough to have a verdict.
if [ ! -f "$REPORT_DIR/regression_report.md" ]; then
    die "the compare step failed before writing a report -- see its error above.
       The measured runs are kept, so fixing it and re-running costs no sweeps."
fi

echo
echo "report: $REPORT_DIR/regression_report.md"
echo "runs:   $BASELINE_CACHE  |  $CURRENT_CACHE"
exit "$STATUS"
