#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for ds_fused_mesh_scatter
# Tests the mesh_scatter communication primitive used in LMHead

set -euo pipefail

# Environment setup
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)}
export PYTHONPATH=${PYTHONPATH:-$(pwd)}
export DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528}
export DEEPSEEK_V3_CACHE=${DEEPSEEK_V3_CACHE:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache}
export MESH_DEVICE=${MESH_DEVICE:-TG}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-/home/models-team/hzhou/tt-metal}

# Test configuration
TEST_PATH="models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_fused_mesh_scatter.py"
LOG_DIR="${TT_METAL_HOME}/logs/fused_mesh_scatter_verification"
TIMEOUT=${TIMEOUT:-600}

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "DeepSeek V3 Fused mesh_scatter Unit Test Verification"
echo "=============================================="
echo "Test path: $TEST_PATH"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a test with description
run_test() {
    local description=$1
    local test_filter=$2
    local log_file="${LOG_DIR}/${3}.log"

    echo "----------------------------------------"
    echo "Running: $description"
    echo "Filter: $test_filter"
    echo "Log: $log_file"
    echo "----------------------------------------"

    timeout $TIMEOUT pytest "$TEST_PATH::test_ds_fused_mesh_scatter" -k "$test_filter" -v 2>&1 | tee "$log_file" || true
    echo ""
}

echo "=========================================="
echo "Step 1: Decode Mode Tests (Row 3 - LMHead default)"
echo "=========================================="

run_test "Decode Row 3 - Eager, Program Cache" \
    "decode and 32 and row_3 and eager and program_cache and not no_program_cache" \
    "decode_row3_eager_pcache"

run_test "Decode Row 3 - Eager, No Program Cache" \
    "decode and 32 and row_3 and eager and no_program_cache" \
    "decode_row3_eager_nopcache"

run_test "Decode Row 3 - Trace, Program Cache" \
    "decode and 32 and row_3 and trace and program_cache and not no_program_cache" \
    "decode_row3_trace_pcache"

echo "=========================================="
echo "Step 2: Decode Mode Tests (All Rows)"
echo "=========================================="

for row in 0 1 2 3; do
    run_test "Decode Row $row - Eager, Program Cache" \
        "decode and 32 and row_$row and eager and program_cache and not no_program_cache" \
        "decode_row${row}_eager_pcache"
done

echo "=========================================="
echo "Step 3: Prefill Mode Tests (Row 3)"
echo "=========================================="

run_test "Prefill 1024 Row 3 - Eager, Program Cache" \
    "prefill and 1024 and row_3 and eager and program_cache and not no_program_cache" \
    "prefill_1024_row3_eager_pcache"

run_test "Prefill 2048 Row 3 - Eager, Program Cache" \
    "prefill and 2048 and row_3 and eager and program_cache and not no_program_cache" \
    "prefill_2048_row3_eager_pcache"

echo "=========================================="
echo "Step 4: Run All Test Combinations (Optional)"
echo "=========================================="
echo "To run all test combinations, use:"
echo "  pytest $TEST_PATH::test_ds_fused_mesh_scatter -v"
echo ""

echo "=========================================="
echo "Step 5: Device Performance Tests"
echo "=========================================="
echo "To run device performance tests, use:"
echo "  pytest $TEST_PATH::test_ds_fused_mesh_scatter_device_perf -v"
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Summary of key test results:"
echo "----------------------------"
for log_file in "$LOG_DIR"/*.log; do
    if [[ -f "$log_file" ]]; then
        filename=$(basename "$log_file" .log)
        if grep -q "PASSED" "$log_file"; then
            echo "  ✓ $filename: PASSED"
        elif grep -q "FAILED" "$log_file"; then
            echo "  ✗ $filename: FAILED"
        elif grep -q "SKIPPED" "$log_file"; then
            echo "  - $filename: SKIPPED"
        else
            echo "  ? $filename: UNKNOWN"
        fi
    fi
done
