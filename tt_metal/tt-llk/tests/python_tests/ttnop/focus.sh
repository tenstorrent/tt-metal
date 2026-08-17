#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Depth runner: one node id, chosen sites and delays, repeated enough times to
# turn a flaky race into a failure rate. Each site also gets a delay-0 control
# that still detours through the cave, so "the jump did it" is ruled out.
#
#   ./focus.sh --sites unpack:3 --nop risc_nop --delays 8,16 \
#       'test_x.py::test_y[params]'
#
# Same poke loop as ci.sh; only the planning and the reporting differ. Every
# flag is an alias for the TTNOP_* variable of the same name, which still works
# on its own when no flag overrides it. See FOCUS.md.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

REPORT_DIR="${TTNOP_REPORT_DIR:-$HERE/reports/focus}"
# One Tensix core per xdist worker; the harness maps gwN to the Nth functional
# core on the card. 1 runs everything in this process.
DEVICE_JOBS="${TTNOP_DEVICE_JOBS:-8}"
NODE_IDS=()

# Flags win over the inherited environment, so a stale export in the shell
# cannot quietly change what a written-down command line claims to sweep.
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

# One node id per run: the report keys on a single case, and an unquoted id whose
# [params] the shell split would otherwise arrive here as several words.
if [[ ${#NODE_IDS[@]} -ne 1 ]]; then
    echo "usage: focus.sh [options] <pytest-node-id>" >&2
    exit 4
fi
NODE_ID="${NODE_IDS[0]}"

# Repeats is what buys the rate: one run of a variant can only say pass or fail,
# ten can say 3/10.
export TTNOP_REPEATS="${TTNOP_REPEATS:-10}"

# Resolve before the cd below, so a relative --report-dir stays under ttnop/.
[[ "$REPORT_DIR" = /* ]] || REPORT_DIR="$HERE/$REPORT_DIR"
mkdir -p "$REPORT_DIR"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
export TTNOP_REPORT_DIR="$REPORT_DIR"

# Records are appended and the markdown is rendered from every one of them, so a
# log left behind by the last run would be folded into this run's report. Keep
# one generation: the previous depth run is often what this one is compared to.
if [[ -f "$REPORT_DIR/failures.jsonl" ]]; then
    mv -f "$REPORT_DIR/failures.jsonl" "$REPORT_DIR/failures.jsonl.prev"
fi
rm -f "$REPORT_DIR/report.md"

# A depth run is one case, so there is nothing for xdist to distribute the usual
# way: `--dist each` hands that case to every worker instead, and the plugin
# splits the variant plan between them (TTNOP_SHARD_VARIANTS) so the cores share
# one sweep rather than each repeating it. Each worker owns its own Tensix, and a
# soft reset after a hang only touches that core, so a wedge stays local.
XDIST_ARGS=()
if [[ "$DEVICE_JOBS" -gt 1 ]]; then
    # Fall back rather than let pytest reject -n after the compile is paid for:
    # xdist comes from tests/requirements.txt and an env can be missing it.
    if python3 -c "import xdist" 2>/dev/null; then
        XDIST_ARGS=(-n "$DEVICE_JOBS" --dist each)
        export TTNOP_SHARD_VARIANTS=1
    else
        echo ">> pytest-xdist not installed; sweeping on one core" >&2
        DEVICE_JOBS=1
    fi
fi

build_scanner
cd "$PYTHON_TESTS"

# Echo the plan before spending the device time, so a log says what it swept.
echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto} repeats=${TTNOP_REPEATS}"
echo ">> case=${NODE_ID}"
echo ">> device_jobs=${DEVICE_JOBS} report=${REPORT_DIR}"

# Build this one variant if the shared tree does not already hold it.
echo ">> [1/2] compiling"
flock "$BUILD_LOCK" python3 -m pytest --compile-producer -q "$NODE_ID"

# The sweep owns the card, cores and all, so hold the device lock for all of it.
exec 9>"$DEVICE_LOCK"
flock 9
echo ">> [2/2] sweeping"
started=$SECONDS
status=0
python3 -m pytest --compile-consumer -p ttnop_plugin -p no:randomly -q \
    "${XDIST_ARGS[@]}" "$NODE_ID" || status=$?
echo ">> timing: sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
