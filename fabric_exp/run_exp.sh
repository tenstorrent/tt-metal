#!/bin/bash
# Run one test_tt_fabric experiment and keep its results under fabric_exp/results/<label>.*
# usage: run_exp.sh <label> <yaml> [ENV=VALUE ...]
#   e.g. run_exp.sh rxdelay50 fabric_exp/exp_base.yaml TT_FABRIC_EXP_RX_DELAY=50
set -u
label=$1
yaml=$2
shift 2
export TT_METAL_HOME=/home/aagarwal/tt-metal TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0
cd "$TT_METAL_HOME"
out=fabric_exp/results
mkdir -p "$out"
rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/latency_results_blackhole.csv \
    "$out/$label.csv" "$out/${label}_latency.csv" "$out/${label}_phase.csv"
echo "=== $label: $* ==="
env "$@" TT_FABRIC_PHASE_PROFILE_CSV="$PWD/$out/${label}_phase.csv" \
    ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$PWD/$yaml" 2>&1 |
    tee "$out/$label.log" | grep --line-buffered -E "Running Test:|FATAL|error:|HUNG"
if [ -f generated/fabric/bandwidth_summary_results_blackhole.csv ]; then
    cp generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
elif [ -f generated/fabric/latency_results_blackhole.csv ]; then
    cp generated/fabric/latency_results_blackhole.csv "$out/${label}_latency.csv"
else
    echo "!!! $label produced no bandwidth or latency summary"
fi
