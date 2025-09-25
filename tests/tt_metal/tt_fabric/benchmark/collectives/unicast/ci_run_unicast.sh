#!/usr/bin/env bash
set -euo pipefail

OUT="${TT_METAL_HOME}/generated/test_reports/unicast"
mkdir -p "$OUT"

COMMON="--sizes 4096,32768,1048576 --iters 150 --page 4096 \
        --tolerance-pct 10 --out-dir $OUT"

# Run 1: 0:0 -> 0:1, recv core 0,0
python tests/tt_metal/tt_fabric/benchmark/collectives/unicast/run_unicast_sweep.py \
  --src 0:0 --dst 0:1 \
  --recv-core 0,0 \
  $COMMON \
  --p50-targets 4096:0.0377505,32768:0.309901,1048576:3.87211 \
  --csv "$OUT/unicast_0to1.csv"

# Run 2: 0:0 -> 0:3, recv core 6,6  (corner-ish)
python tests/tt_metal/tt_fabric/benchmark/collectives/unicast/run_unicast_sweep.py \
  --src 0:0 --dst 0:3 \
  --recv-core 6,6 \
  $COMMON \
  --p50-targets 4096:0.0368183,32768:0.283387,1048576:3.52898 \
  --csv "$OUT/unicast_0to3.csv"
