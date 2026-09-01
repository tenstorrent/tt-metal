#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Repeat one pytest case so a flaky race shows a failure rate. Delay 0 checks
# that the jump itself is not the problem. The flags match the TTNOP_* settings
# described in FOCUS.md.
#
# Defaults to 8 Tensix cores on that one case:
#   pytest-xdist starts 8 workers, each on its own Tensix core. `--dist each`
#   gives every worker the case, then TTNOP_SHARD_VARIANTS splits the NOP plan
#   between those workers.
#
#   ./focus.sh --sites unpack:3 --nop risc_nop --delays 8,16 \
#       'test_x.py::test_y[params]'

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

REPORT_DIR="${TTNOP_REPORT_DIR:-$HERE/reports/focus}"
# Defaults to 8 Tensix on this one case. --device-jobs 1 stays in-process.
DEVICE_JOBS="${TTNOP_DEVICE_JOBS:-8}"
NODE_IDS=()

# Flags beat a leftover export in the shell.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --thread|--threads)  export TTNOP_THREADS="$2";   shift 2 ;;
        --site|--site-mode)  export TTNOP_SITE_MODE="$2"; shift 2 ;;
        --sites)             export TTNOP_SITES="$2";     shift 2 ;;
        --nop|--filler)      export TTNOP_FILLER="$2";    shift 2 ;;
        --delays)            export TTNOP_DELAYS="$2";    shift 2 ;;
        --max-delay)         export TTNOP_MAX_DELAY="$2"; shift 2 ;;
        --repeats)           export TTNOP_REPEATS="$2";   shift 2 ;;
        --device-jobs)       DEVICE_JOBS="$2";            shift 2 ;;
        --no-drift)          export TTNOP_DRIFT=0;        shift ;;
        --verbose)           export TTNOP_VERBOSE=1;      shift ;;
        --report-dir)        REPORT_DIR="$2";             shift 2 ;;
        -*) echo "ttnop: unknown option $1 (see FOCUS.md)" >&2; exit 4 ;;
        *)  NODE_IDS+=("$1"); shift ;;
    esac
done

# Reject an unquoted node id if its parameters split into extra words.
if [[ ${#NODE_IDS[@]} -ne 1 ]]; then
    echo "usage: focus.sh [options] <pytest-node-id>" >&2
    exit 4
fi
NODE_ID="${NODE_IDS[0]}"

# Default 10: one shot is pass/fail, ten is a rate.
export TTNOP_REPEATS="${TTNOP_REPEATS:-10}"

# Resolve before cd so a relative --report-dir stays under ttnop/.
[[ "$REPORT_DIR" = /* ]] || REPORT_DIR="$HERE/$REPORT_DIR"
mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
export TTNOP_REPORT_DIR="$REPORT_DIR"
lock_report_dir "$REPORT_DIR"
reset_report_dir "$REPORT_DIR"

# Split this case's variant plan across the cores (8 unless --device-jobs).
XDIST_ARGS=(-n "$DEVICE_JOBS" --dist each)
export TTNOP_SHARD_VARIANTS=1

build_scanner
cd "$PYTHON_TESTS"

echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto} repeats=${TTNOP_REPEATS}"
echo ">> case=${NODE_ID}"
echo ">> device_jobs=${DEVICE_JOBS} report=${REPORT_DIR}"

# Host-only: build this variant if it isn't already in the shared tree.
echo ">> [1/2] compiling"
flock "$BUILD_LOCK" python3 -m pytest --compile-producer -q \
    "${PYTEST_SIM_ARGS[@]}" "$NODE_ID"

# Don't share the card with another sweep.
exec 9>"$DEVICE_LOCK"
flock 9
echo ">> [2/2] sweeping"
started=$SECONDS
status=0
python3 -m pytest --compile-consumer -p ttnop_plugin -p no:randomly -q \
    "${PYTEST_SIM_ARGS[@]}" "${XDIST_ARGS[@]}" "$NODE_ID" || status=$?
echo ">> timing: sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
