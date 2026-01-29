#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for ds_fused_rms_norm (non-distributed RMSNorm)
# Tests the single ttnn.rms_norm op used for kv_lora_rank/q_lora_rank dimensions

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
TEST_PATH="models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_fused_rms_norm.py"
LOG_DIR="${TT_METAL_HOME}/logs/fused_rms_norm_verification"
TIMEOUT=${TIMEOUT:-600}

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "DeepSeek V3 Fused RMSNorm Unit Test Verification"
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

    timeout $TIMEOUT pytest "$TEST_PATH::test_ds_fused_rms_norm" -k "$test_filter" -v 2>&1 | tee "$log_file" || true
    echo ""
}

echo "=========================================="
echo "Step 1: Decode Mode Tests (kv_lora_rank)"
echo "=========================================="

run_test "Decode kv_lora_rank - Eager, Program Cache, Real Weights" \
    "decode and 32 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and real_weights" \
    "decode_kv_eager_pcache_real"

run_test "Decode kv_lora_rank - Eager, No Program Cache, Real Weights" \
    "decode and 32 and kv_lora_rank_512 and eager and no_program_cache and real_weights" \
    "decode_kv_eager_nopcache_real"

run_test "Decode kv_lora_rank - Trace, Program Cache, Real Weights" \
    "decode and 32 and kv_lora_rank_512 and trace and program_cache and not no_program_cache and real_weights" \
    "decode_kv_trace_pcache_real"

run_test "Decode kv_lora_rank - Eager, Program Cache, Random Weights" \
    "decode and 32 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and random_weights" \
    "decode_kv_eager_pcache_random"

echo "=========================================="
echo "Step 2: Decode Mode Tests (q_lora_rank)"
echo "=========================================="

run_test "Decode q_lora_rank - Eager, Program Cache, Real Weights" \
    "decode and 32 and q_lora_rank_1536 and eager and program_cache and not no_program_cache and real_weights" \
    "decode_q_eager_pcache_real"

run_test "Decode q_lora_rank - Trace, Program Cache, Real Weights" \
    "decode and 32 and q_lora_rank_1536 and trace and program_cache and not no_program_cache and real_weights" \
    "decode_q_trace_pcache_real"

echo "=========================================="
echo "Step 3: Prefill Mode Tests (Short Sequences)"
echo "=========================================="

run_test "Prefill 128 kv_lora_rank - Eager, Program Cache, Real Weights" \
    "prefill and 128 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and real_weights" \
    "prefill_128_kv_eager_pcache_real"

run_test "Prefill 512 q_lora_rank - Eager, Program Cache, Real Weights" \
    "prefill and 512 and q_lora_rank_1536 and eager and program_cache and not no_program_cache and real_weights" \
    "prefill_512_q_eager_pcache_real"

echo "=========================================="
echo "Step 4: Prefill Mode Tests (Long Sequences)"
echo "=========================================="

run_test "Prefill 2048 kv_lora_rank - Eager, Program Cache, Real Weights" \
    "prefill and 2048 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and real_weights" \
    "prefill_2048_kv_eager_pcache_real"

run_test "Prefill 2048 q_lora_rank - Eager, Program Cache, Real Weights" \
    "prefill and 2048 and q_lora_rank_1536 and eager and program_cache and not no_program_cache and real_weights" \
    "prefill_2048_q_eager_pcache_real"

echo "=========================================="
echo "Step 5: Run All Test Combinations (Optional)"
echo "=========================================="
echo "To run all test combinations, use:"
echo "  pytest $TEST_PATH::test_ds_fused_rms_norm -v"
echo ""

echo "=========================================="
echo "Step 6: Device Performance Tests"
echo "=========================================="
echo "To run device performance tests, use:"
echo "  pytest $TEST_PATH::test_ds_fused_rms_norm_device_perf -v"
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
