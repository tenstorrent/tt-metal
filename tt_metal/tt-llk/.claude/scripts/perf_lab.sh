#!/usr/bin/env bash
# One entry point for LLK perf experiments. Queue the whole plan in one command.
#
# It fixes the two failures that keep repeating:
#   * Two runs overlap. The second aborts on a dirty tree, and its cleanup reverts
#     the FIRST run's patches, corrupting a measurement already in flight.
#   * A stale pytest, or a file left patched by a killed run, blocks the next start.
#
# perf_lab takes a lock, waits for anything already running, forces the tree clean,
# then runs each job in order. A second perf_lab started while one is running does
# not fail: it waits its turn. So "run this when that is done" is just a second
# argument, or a second command.
#
# usage: perf_lab.sh [-f] [-p] "JOB" ["JOB" ...]
#   -f   kill a stray pytest instead of waiting for it
#   -p   git pull the branch once the lock is held (safe: nothing else is running)
#
# Each JOB is a shell command run from tests/python_tests, with the scripts
# directory on PATH, so name scripts bare:
#   perf_lab.sh -p \
#     'NOPS="24" RUNS=100 OUT=~/nopwide perf_nop_alignment_sweep.sh' \
#     'OUT=~/nopdisasm perf_nop_disasm_collect.sh' \
#     'perf_nop_alignment_report.py ~/nopwide'
set -uo pipefail

LLK=~/tt-metal/tt_metal/tt-llk
PT=$LLK/tests/python_tests
SRC=$LLK/tests/sources
SCRIPTS=$LLK/.claude/scripts
PATCHED=(perf_math_matmul.py helpers/profiler.py "$SRC/math_matmul_perf.cpp")
LOCK="${LOCK:-$HOME/.llk_perf_lab.lock}"
WAIT_MAX="${WAIT_MAX:-7200}"

FORCE=0; PULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        -f) FORCE=1; shift ;;
        -p) PULL=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown option: $1"; exit 2 ;;
        *)  break ;;
    esac
done
[ $# -gt 0 ] || { sed -n '2,25p' "$0"; exit 2; }

say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
clean_tree() { cd "$PT" && git checkout -- "${PATCHED[@]}" 2>/dev/null; }

# --- one experiment at a time -------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
    say "another perf_lab holds the lock; waiting for it"
    flock -w "$WAIT_MAX" 9 || { say "FATAL: lock still held after ${WAIT_MAX}s"; exit 1; }
fi
say "lock acquired"

# --- wait out anything started outside perf_lab -------------------------------
waited=0
while pgrep -f '[p]ytest' >/dev/null 2>&1; do
    if [ "$FORCE" = 1 ]; then
        say "force: killing stray pytest"
        pkill -f '[p]ytest'; sleep 5; break
    fi
    [ "$waited" -ge "$WAIT_MAX" ] && { say "FATAL: pytest still running after ${waited}s; rerun with -f"; exit 1; }
    [ $((waited % 60)) -eq 0 ] && say "waiting for a pytest already running (${waited}s)"
    sleep 10; waited=$((waited + 10))
done
[ "$waited" -gt 0 ] && say "the earlier run finished after ${waited}s"

# --- always start from a clean tree, whatever the last run left behind --------
clean_tree
cd "$PT"
if ! git diff --quiet -- "${PATCHED[@]}"; then
    say "FATAL: tree still dirty after restore"
    git status --short -- "${PATCHED[@]}"
    exit 1
fi
say "tree clean"

# Armed only now. An abort above must not revert someone else's patches.
trap 'clean_tree; say "tree restored, lock released"' EXIT

if [ "$PULL" = 1 ]; then
    say "pulling"
    ( cd "$LLK" && git pull --ff-only --no-recurse-submodules 2>&1 | tail -3 )
fi

# --- run the queue ------------------------------------------------------------
total=$#; n=0; failed=0
for JOB in "$@"; do
    n=$((n + 1))
    say "job $n/$total: $JOB"
    ( cd "$PT"; PATH="$SCRIPTS:$PATH"; eval "$JOB" )
    rc=$?
    say "job $n/$total rc=$rc"
    clean_tree                       # a crashed job must not block the next one
    if [ "$rc" -ne 0 ]; then
        say "stopping: job $n failed"
        failed=1
        break
    fi
done

[ "$failed" = 0 ] && say "ALL $total JOBS DONE" || say "QUEUE ABORTED"
exit "$failed"
