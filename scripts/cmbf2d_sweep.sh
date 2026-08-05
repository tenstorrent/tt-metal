#!/bin/bash
# cmbf2d_sweep.sh — drive test_combine_fabric2d once per sweep point and collect the per-point
# bandwidth summary. One pytest invocation per point (a device open each time) because the token counts
# and token size are compile-time kernel args, so each point is a fresh JIT build anyway.
#
# Usage:
#   scripts/cmbf2d_sweep.sh tokens 32 100 256 1000
#   scripts/cmbf2d_sweep.sh label <tag>          # single point, current env, tagged <tag>
#
# Env passthrough: CMBF2D_TOKENS / CMBF2D_TOKEN_BYTES set the axes
# that are NOT being swept. Results land in generated/cmbf2d/bwinfo_<tag>.txt; a one-line summary per
# point is appended to generated/cmbf2d/sweep_<axis>.log.

set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TEST="models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_prefill_combine.py::test_combine_fabric2d"
FILTER="fabric2d-torus-xy-8x4-2link"

AXIS="$1"
shift
LOG="generated/cmbf2d/sweep_${AXIS}.log"
mkdir -p generated/cmbf2d

run_point() {
    local tag="$1"
    echo "=== [$(date +%H:%M:%S)] point ${tag}: tokens=${CMBF2D_TOKENS:-100} token=${CMBF2D_TOKEN_BYTES:-14336} bump=${CMBF2D_FWD_BUMP:-32} order=${CMBF2D_ORDER:-1} stall=${CMBF2D_STALL:-0}"
    CMBF2D_TAG="$tag" CMBF2D_FWD_BUMP="${CMBF2D_FWD_BUMP:-32}" CMBF2D_ORDER="${CMBF2D_ORDER:-1}" scripts/run_safe_pytest.sh "$TEST" -k "$FILTER" -s \
        >"generated/cmbf2d/run_${tag}.log" 2>&1
    local rc=$?
    local bw="generated/cmbf2d/bwinfo_${tag}.txt"
    if [[ -f "$bw" ]]; then
        # The file's own summary lines. Accuracy is not among them: the test asserts it, so rc=0
        # already means it passed.
        local summary send shares nvalid
        summary=$(grep "SLOWEST producer" "$bw")
        send=$(grep "per-producer sGB/s" "$bw")
        shares=$(grep "mean send-window shares" "$bw")
        nvalid=$(grep -c "^ *[0-9]" "$bw")
        {
            echo "${tag} rc=${rc} workers=${nvalid} ${summary#\# per-producer GB/s: }"
            echo "    send ${send#\# per-producer sGB/s: }"
            echo "    ${shares#\# mean send-window }"
        } | tee -a "$LOG"
    else
        echo "${tag} rc=${rc} NO REPORT (see generated/cmbf2d/run_${tag}.log)" | tee -a "$LOG"
    fi
    return 0
}

case "$AXIS" in
    tokens)
        for n in "$@"; do
            CMBF2D_TOKENS="$n" run_point "tok${n}"
        done
        ;;
    token)
        for n in "$@"; do
            CMBF2D_TOKEN_BYTES="$n" run_point "tokb${n}"
        done
        ;;
    order)
        for n in "$@"; do
            CMBF2D_ORDER="$n" run_point "order${n}"
        done
        ;;
    bump)
        for n in "$@"; do
            CMBF2D_FWD_BUMP="$n" run_point "bump${n}"
        done
        ;;
    label)
        run_point "$1"
        ;;
    *)
        echo "usage: $0 {tokens|token|bump|order|label} <values...>"
        exit 3
        ;;
esac

echo "--- sweep ${AXIS} done; summary in ${LOG}"
cat "$LOG"
