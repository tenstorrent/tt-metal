#!/bin/bash
# Knee runs of the streaming profiler: full Tensix grid, uniform zone rate per lane, each run sized to about
# <target_ms> of producer time. Run from the tt-metal root.
# Usage: knee.sh <outdir> <target_ms> <reps> <delay>...
set -u
OUT=$1; TARGET_MS=$2; REPS=$3; shift 3
BIN=${ZONES_BIN:-./build_Release/programming_examples/test_streaming_profiler_zones}
mkdir -p "$OUT"
for D in "$@"; do
  ITERS=$(awk -v d="$D" -v t="$TARGET_MS" 'BEGIN { z = 86.7 + d * 10.0 / 1.35; printf "%d", t * 1e6 / (10 * z) }')
  for REP in $(seq 1 "$REPS"); do
    LOG="$OUT/d${D}_r${REP}.log"
    TT_METAL_STREAMING_PROFILER=1 timeout 900 "$BIN" --gx 0 --gy 0 --iters "$ITERS" --delay "$D" >"$LOG" 2>&1
    echo "d=$D r=$REP rc=$? | $(grep -oE 'workload done in [0-9.]+ ms' "$LOG") | $(grep -oE 'Device 0: [0-9]+ profiler stalls' "$LOG") | $(grep -oE 'capture: [0-9]+ frames, [0-9.]+ MB' "$LOG") | missing=$(grep -c 'not in the capture' "$LOG")"
  done
done
