#!/bin/bash
# The profiler's zones run under a steady copy load (DriscService.SteadyLoad) from a second process, for each drain
# hold setting N:M (copy turns between holds : sweeps per hold) and load (none, local, d2h). Prints one row per run
# (sweep_row.py): drainer stalls, and the copy GB/s and 64 B probe latency before the profiler and during its workload.
# Run from the tt-metal root.
# Usage: sweep.sh <outdir> <delay> <reps> "<N:M ...>" "<load ...>"   (TARGET_MS sets the producer time, default 200)
set -u
OUT=$1; D=$2; REPS=$3; CFGS=$4; LOADS=$5
HERE=$(cd "$(dirname "$0")" && pwd)
ZONES=${ZONES_BIN:-./build_Release/programming_examples/test_streaming_profiler_zones}
UT=${UT_BIN:-./build_Release/test/tt_metal/unit_tests_drisc_service}
ITERS=$(awk -v d="$D" -v t="${TARGET_MS:-200}" 'BEGIN { z = 86.7 + d * 10.0 / 1.35; printf "%d", t * 1e6 / (10 * z) }')
mkdir -p "$OUT"
stamp() { while IFS= read -r l; do printf "%s %s\n" "$(date +%s%N)" "$l"; done; }
for CFG in $CFGS; do N=${CFG%%:*}; M=${CFG##*:}
for LOAD in $LOADS; do for R in $(seq 1 "$REPS"); do
  TAG="d${D}_n${N}_m${M}_${LOAD}_r${R}"; Z="$OUT/$TAG.zones"; L="$OUT/$TAG.load"
  LP=""
  if [ "$LOAD" != none ]; then
    DRISC_SERVICE_LOAD_SECONDS=${LOAD_SECONDS:-25} DRISC_SERVICE_LOAD_OP=$LOAD DRISC_SERVICE_LOAD_REPORT=$L \
      "$UT" --gtest_filter=DriscService.SteadyLoad > "$OUT/$TAG.ut" 2>&1 &
    LP=$!; sleep 2
  fi
  TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_TURNS=$N TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_SWEEPS=$M \
    timeout 600 stdbuf -oL -eL "$ZONES" --gx 0 --gy 0 --iters "$ITERS" --delay "$D" 2>&1 | stamp > "$Z"
  [ -n "$LP" ] && wait $LP
  python3 "$HERE/sweep_row.py" "$TAG" "$Z" "$L"
done; done; done
