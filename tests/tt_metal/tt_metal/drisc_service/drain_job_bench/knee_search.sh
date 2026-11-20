#!/bin/bash
# For each copy load and drain hold setting N (M = 1), raises the profiler's delay until all reps of a delay show 0
# device stalls (the knee), then runs one delay above it to confirm. Run from the tt-metal root.
# Usage: knee_search.sh <outdir> <reps> "<load ...>" "<N ...>" "<delay ...ascending>"
set -u
OUT=$1; REPS=$2; LOADS=$3; NS=$4; DELAYS=$5
HERE=$(cd "$(dirname "$0")" && pwd)
for LOAD in $LOADS; do for N in $NS; do
  knee=""
  for D in $DELAYS; do
    rows=$("$HERE/sweep.sh" "$OUT/${LOAD}_n${N}" "$D" "$REPS" "$N:1" "$LOAD")
    echo "$rows"
    nonzero=$(echo "$rows" | grep -cv " stalls=0 ")
    if [ -n "$knee" ]; then
      echo "KNEE load=$LOAD N=$N: first all-zero delay $knee; confirm at $D: $((REPS - nonzero))/$REPS zero"
      break
    fi
    [ "$nonzero" -eq 0 ] && knee=$D
  done
  [ -z "$knee" ] && echo "KNEE load=$LOAD N=$N: no all-zero delay up to the last one tried"
  [ -n "$knee" ] && [ "$D" = "$knee" ] && echo "KNEE load=$LOAD N=$N: first all-zero delay $knee (no higher delay to confirm)"
done; done
