#!/bin/bash
# Byte-verified copies (DriscService.Stress: rolling client children, every copy checked) on the seats the drain
# jobs share, while profiler knee runs come and go, for each drain hold setting N (M = 1). Stress outlives the knee
# runs, so its end-of-test check (every service kernel stopped once the last client left) still applies.
# Run from the tt-metal root.
# Usage: stress_with_knee.sh <outdir> <stress_seconds> <reps> "<N ...>" "<delay ...>"
set -u
OUT=$1; SECS=$2; REPS=$3; NS=$4; DELAYS=$5
HERE=$(cd "$(dirname "$0")" && pwd)
UT=${UT_BIN:-./build_Release/test/tt_metal/unit_tests_drisc_service}
mkdir -p "$OUT"
for N in $NS; do
  DRISC_SERVICE_STRESS_SECONDS=$SECS DRISC_SERVICE_STRESS_REPORT="$OUT/stress_n$N.report" \
    "$UT" --gtest_filter=DriscService.Stress > "$OUT/stress_n$N.log" 2>&1 &
  SP=$!; sleep 5
  TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_TURNS=$N TT_METAL_STREAMING_PROFILER_DRAIN_HOLD_SWEEPS=1 \
    "$HERE/knee.sh" "$OUT/knee_n$N" 200 "$REPS" $DELAYS
  wait $SP; RC=$?
  echo "STRESS N=$N rc=$RC $(grep -E '^\[ +(OK|FAILED) +\] DriscService.Stress' "$OUT/stress_n$N.log") $(tr '\n' ' ' < "$OUT/stress_n$N.report" 2>/dev/null)"
  grep -iE "mismatch|Failure|error:" "$OUT/stress_n$N.log" | head -5
done
