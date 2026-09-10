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
#
# --metal sweeps a ttnn op test instead of an LLK kernel test; the node id is
# then written from the repo root.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

REPORT_DIR="${TTNOP_REPORT_DIR:-$HERE/reports/focus}"
# Defaults to 8 Tensix on this one case. --device-jobs 1 stays in-process.
DEVICE_JOBS="${TTNOP_DEVICE_JOBS:-8}"
METAL=0
NODE_IDS=()

# Flags beat a leftover export in the shell.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --thread|--threads)  export TTNOP_THREADS="$2";   shift 2 ;;
        --site|--site-mode)  export TTNOP_SITE_MODE="$2"; shift 2 ;;
        --sites)             export TTNOP_SITES="$2";     shift 2 ;;
        --nop|--filler)      export TTNOP_FILLER="$2";    shift 2 ;;
        --enable-unpacr-nop) export TTNOP_ENABLE_UNPACR_NOP=1; shift ;;
        --delays)            export TTNOP_DELAYS="$2";    shift 2 ;;
        --max-delay)         export TTNOP_MAX_DELAY="$2"; shift 2 ;;
        --repeats)           export TTNOP_REPEATS="$2";   shift 2 ;;
        --device-jobs)       DEVICE_JOBS="$2";            shift 2 ;;
        --metal)             METAL=1;                     shift ;;
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
setup_report_dir
reset_report_dir "$REPORT_DIR"

# One image serves every core running the op, so a ttnn test occupies the whole
# grid and a second worker has no core of its own: the plan stays in one process.
if [[ "$METAL" == 1 ]]; then
    metal_env
    DEVICE_JOBS=1
fi

if [[ "$CHIP_ARCH" == "quasar" ]]; then
    DEVICE_JOBS=1
fi

# Split this case's variant plan across the cores (8 unless --device-jobs).
XDIST_ARGS=()
[[ "$DEVICE_JOBS" -gt 1 ]] && XDIST_ARGS=(-n "$DEVICE_JOBS" --dist each)
export TTNOP_SHARD_VARIANTS=1

build_scanner
cd "$TESTS_ROOT"

echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto}" \
     "unpacr_nop=${TTNOP_ENABLE_UNPACR_NOP:-0} repeats=${TTNOP_REPEATS}"
echo ">> case=${NODE_ID}"
echo ">> device_jobs=${DEVICE_JOBS} report=${REPORT_DIR}"
if [[ "$METAL" == 1 ]]; then
    echo ">> metal kernel=${TTNOP_METAL_KERNEL:-<most recently loaded>} arch=${CHIP_ARCH} simulator=${TT_METAL_SIMULATOR:-<none>}"
fi

# Build this one variant if the shared tree does not already hold it. Metal JITs
# its own kernels on the first launch and has no producer pass.
if [[ "$METAL" == 0 ]]; then
    echo ">> [1/2] compiling"
    flock "$BUILD_LOCK" python3 -m pytest --compile-producer -q \
        "${PYTEST_SIM_ARGS[@]}" "$NODE_ID"
fi

# Don't share the card with another sweep.
exec 9>"$DEVICE_LOCK"
flock 9
echo ">> [2/2] sweeping"
started=$SECONDS
status=0
python3 -m pytest "${CONSUMER_ARGS[@]}" -p ttnop_plugin -p no:randomly -q \
    "${PYTEST_SIM_ARGS[@]}" "${XDIST_ARGS[@]}" "$NODE_ID" || status=$?
echo ">> timing: sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
