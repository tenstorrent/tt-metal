#!/bin/bash
# Tracy device-time profile of one MiniMax-H3 transformer block, dense vs VSA, at 5/10/15 s 768p on a
# 4x8 Blackhole Galaxy. One profiled pytest per (mode, duration): a multi-param profiled run only keeps
# the first param's ops.
#
#   MODES="dense vsa" DURS="5 10 15" OUT=/path/to/dir models/tt_dit/models/transformers/minimax_h3/scripts/profile_block.sh
#
# Writes <OUT>/<mode>_<dur>s.csv (tt-perf-report input) and <OUT>/<mode>_<dur>s.report.txt.
# Note the block-level numbers are for a single warm block; sustained runs power-throttle the dense
# block (see VSA_STREAM_DESIGN.md 8), so compare dense vs VSA end to end as well.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../../../../.." && pwd)
cd "$REPO"
export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$REPO/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto}"
OUT="${OUT:-$REPO/generated/vsa_block_perf}"
mkdir -p "$OUT"
REPORTS="$REPO/generated/profiler/reports"
run_one() {
    local label="$1" test_id="$2"
    echo "=== PROFILING $label ==="
    local before
    before=$(ls -t "$REPORTS"/*/ops_perf_results*.csv 2>/dev/null | head -1)
    ./scripts/run_safe_pytest.sh --profile "$test_id" > "$OUT/$label.pytest.log" 2>&1
    local csv
    csv=$(ls -t "$REPORTS"/*/ops_perf_results*.csv 2>/dev/null | head -1)
    if [[ -z "$csv" || "$csv" == "$before" ]]; then
        echo "$label: NO CSV PRODUCED (see $OUT/$label.pytest.log)"
        return 1
    fi
    cp "$csv" "$OUT/$label.csv"
    ./python_env/bin/tt-perf-report "$OUT/$label.csv" --start-signpost start --end-signpost stop \
        > "$OUT/$label.report.txt" 2>&1
    echo "$label: csv=$OUT/$label.csv"
    tail -5 "$OUT/$label.report.txt"
}
DENSE=models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf
VSA=models/tt_dit/tests/models/minimax_h3/test_vsa_performance_minimax_h3.py::test_minimax_h3_vsa_block_perf
for dur in ${DURS:-5 10 15}; do
    for mode in ${MODES:-dense vsa}; do
        if [[ "$mode" == dense ]]; then
            run_one "dense_${dur}s" "$DENSE[blackhole-sp_sim1-${dur}s_768p-4x8sp1tp0nl2_ring_is_fsdp0]"
        else
            run_one "vsa_${dur}s" "$VSA[blackhole-${dur}s_768p-4x8sp1tp0nl2_ring_is_fsdp0]"
        fi
    done
done
echo "PERF_SWEEP_DONE"
