#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Breadth runner: compile the suite once, then sweep every planned
# (thread, site, delay) exactly once behind a clean baseline pass.
#
#   ./ci.sh --test test_eltwise_unary_datacopy.py [--k EXPR]
#           [--markers 'not perf and not nightly and not accuracy and not quasar']
#           [--splits N --group G] [--report-dir DIR] [--jobs N]
#           [--device-jobs N] [--collect-to FILE] [--nodeids FILE]
#
# --test . (from python_tests/) runs the whole suite. --splits/--group divide
# one suite across machines, pytest-split style. Give each machine its own
# --report-dir.
#
# Env (see README): TTNOP_DELAYS TTNOP_THREADS TTNOP_SITE_MODE TTNOP_FILLER
# Markers also accepted via PYTEST_MARKERS / TTNOP_MARKERS (CI sets PYTEST_MARKERS).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TESTS=()
K=""
MARKERS="${TTNOP_MARKERS:-${PYTEST_MARKERS:-}}"
SPLITS=""
GROUP=""
JOBS="${TTNOP_JOBS:-15}"
# One Tensix core per xdist worker (harness maps gwN -> core N/8,N%8).
DEVICE_JOBS="${TTNOP_DEVICE_JOBS:-8}"
REPORT_DIR="${TTNOP_REPORT_DIR:-$HERE/reports}"
COLLECT_TO=""
NODEIDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TESTS+=("$2"); shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --markers) MARKERS="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --device-jobs) DEVICE_JOBS="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --collect-to) COLLECT_TO="$2"; shift 2 ;;
        --nodeids) NODEIDS="$2"; shift 2 ;;
        *) echo "ttnop: unknown option $1" >&2; exit 4 ;;
    esac
done

if [[ ${#TESTS[@]} -eq 0 && -z "$NODEIDS" ]]; then
    echo "ttnop: need --test FILE or --nodeids FILE" >&2
    exit 4
fi

# Resolve while we are still in the caller's directory: sweep.py resolves
# TTNOP_REPORT_DIR against the pytest process's cwd, which the cd below makes
# $PYTHON_TESTS, so a relative --report-dir would land a level up from where
# the caller (and CI's artifact glob) expects it.
mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
export TTNOP_REPORT_DIR="$REPORT_DIR"

# Where workers publish progress and the supervisor keeps its resume list. Wiped
# per invocation: a stale done-log would silently skip the whole sweep.
STATE_DIR="${TTNOP_STATE_DIR:-$REPORT_DIR/.state}"
rm -rf "$STATE_DIR"
mkdir -p "$STATE_DIR"
export TTNOP_STATE_DIR="$STATE_DIR"

build_scanner
cd "$PYTHON_TESTS"

SPLIT_ARGS=()
[[ -n "$SPLITS" ]] && SPLIT_ARGS+=(--splits "$SPLITS" --group "${GROUP:-1}")
FILTER_ARGS=()
[[ -n "$K" ]] && FILTER_ARGS+=(-k "$K")
[[ -n "$MARKERS" ]] && FILTER_ARGS+=(-m "$MARKERS")
XDIST_ARGS=()
[[ "$DEVICE_JOBS" -gt 1 ]] && XDIST_ARGS=(-n "$DEVICE_JOBS")
# Per-test reporting is priced per case, and a sweep is tens of thousands of them:
# in run 31714492632 the compile phase alone streamed ~82k lines in six minutes,
# all of it "SKIPPED (compiling)". pytest-sugar prints two full lines per test
# under xdist where the default reporter prints one character; classic style drops
# the progress counter that is redrawn after every one of them; and log_cli (on by
# default in pytest.ini) replays every loguru record the harness emits. None of it
# survives as something a human reads afterwards. Progress comes from supervise.py
# instead, which counts the done-log every few minutes.
#
# run_ttsim_regression.sh warns against these overrides, but that applies to its
# --forked run, where they emptied pytest-forked's child output out of the junit
# <system-out> it reports from. The sweep is xdist-only and its findings come from
# report.append() in-process, so nothing here reads back through capture.
QUIET_ARGS=(-p no:sugar -o console_output_style=classic -o log_cli=false)

echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto}"
echo ">> test=${TESTS[*]:-$NODEIDS} compile_jobs=${JOBS} device_jobs=${DEVICE_JOBS}"
[[ -z "$MARKERS" ]] || echo ">> markers=${MARKERS}"
[[ -z "$SPLITS" ]] || echo ">> splits=${SPLITS} group=${GROUP:-1}"

compile_s=0
if [[ -z "$NODEIDS" ]]; then
    # Compile on CPUs. The producer wipes the shared build tree on entry, so hold an
    # exclusive lock or a parallel worker will delete artifacts out from under itself.
    echo ">> [1/3] compiling"
    started=$SECONDS
    flock "$BUILD_LOCK" \
        python3 -m pytest --compile-producer -n "$JOBS" -q \
            "${QUIET_ARGS[@]}" "${FILTER_ARGS[@]}" "${TESTS[@]}"
    compile_s=$((SECONDS - started))

    echo ">> [2/3] collecting"
    NODEIDS="$(mktemp /tmp/ttnop-nodeids-XXXXXX)"
    trap '[[ -n "${COLLECT_TO:-}" ]] || rm -f "$NODEIDS"' EXIT
    # Quiet here is about the grep as much as the volume: it keeps stdout to bare
    # node ids, so nothing decorated can sneak past the '::' filter into the list.
    python3 -m pytest --collect-only -q --compile-consumer \
        "${QUIET_ARGS[@]}" "${SPLIT_ARGS[@]}" "${FILTER_ARGS[@]}" "${TESTS[@]}" \
        | grep '::' > "$NODEIDS" || true
    echo ">> $(grep -c . "$NODEIDS") case(s)"
fi

if [[ -n "$COLLECT_TO" ]]; then
    cp "$NODEIDS" "$COLLECT_TO"
    echo ">> node ids -> $COLLECT_TO"
    exit 0
fi

# One sweep at a time per arch: these tests own the whole Tensix.
exec 9>"$DEVICE_LOCK"
flock 9
echo ">> [3/3] sweeping"
started=$SECONDS
# A found race turns the case red, so a non-zero exit is a result, not a crash:
# report the timing either way and pass the code on. The supervisor rides out a
# wedged card by resetting and resuming, and reports that separately by exiting
# 75, so a caller can tell "found races" from "the card stopped answering".
status=0
supervise_nodeids "$NODEIDS" --compile-consumer \
    "${QUIET_ARGS[@]}" "${XDIST_ARGS[@]}" || status=$?
echo ">> timing: compile=${compile_s}s sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
