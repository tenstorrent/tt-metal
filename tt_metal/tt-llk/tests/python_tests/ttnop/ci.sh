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

export TTNOP_REPORT_DIR="$REPORT_DIR"
build_scanner
cd "$PYTHON_TESTS"

SPLIT_ARGS=()
[[ -n "$SPLITS" ]] && SPLIT_ARGS+=(--splits "$SPLITS" --group "${GROUP:-1}")
FILTER_ARGS=()
[[ -n "$K" ]] && FILTER_ARGS+=(-k "$K")
[[ -n "$MARKERS" ]] && FILTER_ARGS+=(-m "$MARKERS")
XDIST_ARGS=()
[[ "$DEVICE_JOBS" -gt 1 ]] && XDIST_ARGS=(-n "$DEVICE_JOBS")

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
        python3 -m pytest --compile-producer -n "$JOBS" -q "${FILTER_ARGS[@]}" "${TESTS[@]}"
    compile_s=$((SECONDS - started))

    echo ">> [2/3] collecting"
    NODEIDS="$(mktemp /tmp/ttnop-nodeids-XXXXXX)"
    trap '[[ -n "${COLLECT_TO:-}" ]] || rm -f "$NODEIDS"' EXIT
    python3 -m pytest --collect-only -q --compile-consumer \
        "${SPLIT_ARGS[@]}" "${FILTER_ARGS[@]}" "${TESTS[@]}" \
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
# report the timing either way and pass the code on.
status=0
run_nodeids "$NODEIDS" --compile-consumer "${XDIST_ARGS[@]}" || status=$?
echo ">> timing: compile=${compile_s}s sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
