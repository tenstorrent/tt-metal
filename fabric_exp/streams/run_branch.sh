#!/bin/bash
# Run one test_tt_fabric experiment from the fabric-test-streams branch worktree (stock router) and keep
# its results under fabric_exp/streams/results/<label>.*
# usage: run_branch.sh <label> <yaml> [ENV=VALUE ...]
set -u
label=$1
yaml=$(realpath "$2")
shift 2
out=/home/aagarwal/tt-metal/fabric_exp/streams/results
export TT_METAL_HOME=/home/aagarwal/tt-metal-streams TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0
cd "$TT_METAL_HOME"
mkdir -p "$out"
rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/bandwidth_results_blackhole.csv \
    "$out/$label.csv" "$out/${label}_links.csv"
echo "=== $label: $* ==="
env "$@" TT_FABRIC_NO_BW_TELEMETRY=1 \
    ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config "$yaml" 2>&1 |
    tee "$out/$label.log" | grep --line-buffered -E "Running Test:|FATAL|error:|HUNG|PASS|FAIL"
[ -f generated/fabric/bandwidth_summary_results_blackhole.csv ] && cp generated/fabric/bandwidth_summary_results_blackhole.csv "$out/$label.csv"
[ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv "$out/${label}_links.csv"
exit 0
