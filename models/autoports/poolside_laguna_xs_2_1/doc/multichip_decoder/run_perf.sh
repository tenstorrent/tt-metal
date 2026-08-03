#!/usr/bin/env bash
# Multichip perf + tt-perf-report + watcher runner (run steps ONE AT A TIME; watcher separate
# from Tracy). Env: installed self-consistent tree, non-repo cwd.
set -euo pipefail
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal
cd /tmp
ROOT=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
ART=$ROOT/doc/multichip_decoder
T=$ROOT/tests

# 1) warmed single-chip baseline + multichip latency/speedup/efficiency (clean, non-Tracy)
for L in 0 1 4; do
  python $T/perf_trace_mc.py $L 512 50 both
done

# 2) Tracy device-op tables (per layer kind): decode + prefill signposted -> tt-perf-report
for L in 0 1 4; do
  OUT=$ART/tracy/layer$L
  mkdir -p $OUT
  python -m tracy -r -p -v -o $OUT $T/perf_trace_mc.py $L 512 20 mc
  # ops.csv name may vary; find the produced csv
  CSV=$(ls -t $OUT/*.csv 2>/dev/null | head -1 || true)
done
