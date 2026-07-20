#!/bin/bash
# cmbf2d_sweep.sh — drive test_combine_fabric2d once per sweep point and collect the per-point
# bandwidth summary. One pytest invocation per point (a device open each time) because num_tokens /
# num_slots / chunk_size_bytes are compile-time kernel args, so each point is a fresh JIT build anyway.
#
# Usage:
#   scripts/cmbf2d_sweep.sh tokens 10 32 100 300 1000 3000 10000
#   CMBF2D_NUM_TOKENS=3000 scripts/cmbf2d_sweep.sh slots 8 16 32 64
#   scripts/cmbf2d_sweep.sh label <tag>          # single point, current env, tagged <tag>
#
# Env passthrough: CMBF2D_NUM_TOKENS / CMBF2D_NUM_SLOTS / CMBF2D_CHUNK_BYTES set the axes that are
# NOT being swept. Results land in generated/cmbf2d/bwinfo_<tag>.txt; a one-line summary per point is
# appended to generated/cmbf2d/sweep_<axis>.log.

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
    echo "=== [$(date +%H:%M:%S)] point ${tag}: tokens=${CMBF2D_NUM_TOKENS:-100} slots=${CMBF2D_NUM_SLOTS:-32} chunk=${CMBF2D_CHUNK_BYTES:-14336} stall=${CMBF2D_STALL:-0} variant=${CMBF2D_VARIANT:-0}"
    CMBF2D_TAG="$tag" scripts/run_safe_pytest.sh "$TEST" -k "$FILTER" -s \
        >"generated/cmbf2d/run_${tag}.log" 2>&1
    local rc=$?
    local bw="generated/cmbf2d/bwinfo_${tag}.txt"
    if [[ -f "$bw" ]]; then
        # The file's own summary lines: end-to-end GB/s, push-rate sGB/s, and the stall shares.
        local summary send shares nvalid
        summary=$(grep "per-producer GB/s" "$bw")
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
            CMBF2D_NUM_TOKENS="$n" run_point "tok${n}"
        done
        ;;
    slots)
        for n in "$@"; do
            CMBF2D_NUM_SLOTS="$n" run_point "slots${n}"
        done
        ;;
    chunk)
        for n in "$@"; do
            CMBF2D_CHUNK_BYTES="$n" run_point "chunk${n}"
        done
        ;;
    variant)
        for n in "$@"; do
            CMBF2D_VARIANT="$n" run_point "var${n}"
        done
        ;;
    label)
        run_point "$1"
        ;;
    *)
        echo "usage: $0 {tokens|slots|chunk|label} <values...>"
        exit 3
        ;;
esac

echo "--- sweep ${AXIS} done; summary in ${LOG}"
cat "$LOG"
