#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# CI Test Script for DeepSeek V3 Long Sequence (128K) Tests
# Purpose: Identify ops that may run out of memory with long sequences
#
# Tests all ops with:
#   - Prefill mode
#   - 128K sequence length (131072)
#   - Program cache enabled
#   - Trace mode enabled

set -uo pipefail

# Environment setup
export ARCH_NAME=${ARCH_NAME:-wormhole_b0}
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)}
export DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528}
export DEEPSEEK_V3_CACHE=${DEEPSEEK_V3_CACHE:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache}
export MESH_DEVICE=${MESH_DEVICE:-TG}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME}}

# Enable long sequence tests
export DEEPSEEK_V3_LONG_SEQ_TESTS=1

# Activate Python virtual environment (check both common locations)
if [ -f "${TT_METAL_HOME}/python_env/bin/activate" ]; then
    echo "Activating Python virtual environment from ${TT_METAL_HOME}/python_env..."
    source "${TT_METAL_HOME}/python_env/bin/activate"
elif [ -f "${TT_METAL_HOME}/build/python_env/bin/activate" ]; then
    echo "Activating Python virtual environment from ${TT_METAL_HOME}/build/python_env..."
    source "${TT_METAL_HOME}/build/python_env/bin/activate"
else
    echo "ERROR: Python virtual environment not found."
    echo "Checked: ${TT_METAL_HOME}/python_env and ${TT_METAL_HOME}/build/python_env"
    echo "Please activate the environment before running this script."
    exit 1
fi

# Set PYTHONPATH after venv activation to ensure ttnn is found
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"

# Debug: Show environment
echo "TT_METAL_HOME: ${TT_METAL_HOME}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "Python: $(which python)"

# Test configuration
BASE_PATH="models/demos/deepseek_v3/tests/fused_op_unit_tests"
LOG_DIR="${TT_METAL_HOME}/logs/ci_long_seq_$(date +%Y%m%d_%H%M%S)"
TIMEOUT=${TIMEOUT:-3600}  # 60 minutes per test (long sequences take longer)

# Create log directory
mkdir -p "$LOG_DIR"

# Track results
PASSED=0
FAILED=0
OOM=0

echo "=============================================="
echo "DeepSeek V3 Long Sequence (128K) Tests"
echo "=============================================="
echo "Testing: Prefill 131072 with trace + program_cache"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a single test and track result
run_test() {
    local test_name=$1
    local test_path=$2
    local test_func=$3
    local test_filter=$4
    local log_file="${LOG_DIR}/${test_name}.log"

    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "Path: $test_path::$test_func"
    echo "Filter: $test_filter"
    echo "Log: $log_file"
    echo "----------------------------------------"

    local start_time=$(date +%s)

    if timeout $TIMEOUT pytest "$test_path::$test_func" -k "$test_filter" -v 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if grep -q "PASSED" "$log_file"; then
            echo "✓ $test_name: PASSED (${duration}s)"
            PASSED=$((PASSED + 1))
        elif grep -q "SKIPPED" "$log_file"; then
            echo "- $test_name: SKIPPED (${duration}s)"
        else
            echo "? $test_name: UNKNOWN (${duration}s)"
        fi
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        # Check for OOM errors
        if grep -qi "out of memory\|OOM\|ENOMEM\|memory allocation failed\|cannot allocate" "$log_file"; then
            echo "💥 $test_name: OOM (${duration}s)"
            OOM=$((OOM + 1))
        elif grep -q "SKIPPED" "$log_file"; then
            echo "- $test_name: SKIPPED (${duration}s)"
        else
            echo "✗ $test_name: FAILED (${duration}s)"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
}

# Common filter for all tests: prefill 131072 with trace and program_cache
LONG_SEQ_FILTER="prefill and 131072 and trace and program_cache and not no_program_cache"

# ============================================
# EMBEDDING TESTS
# ============================================
echo "=========================================="
echo "EMBEDDING - 128K Prefill Trace Test"
echo "=========================================="

run_test "embedding_prefill_128k_trace" \
    "${BASE_PATH}/embedding/test_ds_embedding.py" \
    "test_ds_embedding" \
    "${LONG_SEQ_FILTER}"

# ============================================
# EMBEDDING ALL GATHER TESTS
# ============================================
echo "=========================================="
echo "EMBEDDING ALL GATHER - 128K Prefill Trace Test"
echo "=========================================="

run_test "embedding_all_gather_prefill_128k_trace" \
    "${BASE_PATH}/embedding/test_ds_all_gather_embedding.py" \
    "test_ds_all_gather_embedding" \
    "${LONG_SEQ_FILTER}"

# ============================================
# LM HEAD TESTS
# ============================================
echo "=========================================="
echo "LM HEAD - 128K Prefill Trace Test"
echo "=========================================="

run_test "lm_head_prefill_128k_trace" \
    "${BASE_PATH}/lm_head/test_ds_lm_head.py" \
    "test_ds_lm_head" \
    "${LONG_SEQ_FILTER} and real_weights"

# ============================================
# RMS NORM TESTS
# ============================================
echo "=========================================="
echo "RMS NORM - 128K Prefill Trace Test"
echo "=========================================="

run_test "rms_norm_prefill_128k_trace" \
    "${BASE_PATH}/rms_norm/test_ds_rms_norm.py" \
    "test_ds_rms_norm" \
    "${LONG_SEQ_FILTER} and kv_lora_rank_512 and real_weights"

# ============================================
# DISTRIBUTED NORM TESTS
# ============================================
echo "=========================================="
echo "DISTRIBUTED NORM - 128K Prefill Trace Test"
echo "=========================================="

run_test "distributed_norm_prefill_128k_trace" \
    "${BASE_PATH}/rms_norm/test_ds_distributed_norm.py" \
    "test_ds_distributed_norm" \
    "${LONG_SEQ_FILTER} and real_weights"

# ============================================
# MLP FF1/3 TESTS
# ============================================
echo "=========================================="
echo "MLP FF1/3 - 128K Prefill Trace Test"
echo "=========================================="

run_test "ff1_3_prefill_128k_trace" \
    "${BASE_PATH}/mlp/test_ds_ff1_3.py" \
    "test_ds_ff1_3" \
    "${LONG_SEQ_FILTER} and real_weights"

# ============================================
# MLP FF2 TESTS
# ============================================
echo "=========================================="
echo "MLP FF2 - 128K Prefill Trace Test"
echo "=========================================="

run_test "ff2_prefill_128k_trace" \
    "${BASE_PATH}/mlp/test_ds_ff2.py" \
    "test_ds_ff2" \
    "${LONG_SEQ_FILTER} and real_weights"

# ============================================
# MLP MUL TESTS
# ============================================
echo "=========================================="
echo "MLP MUL - 128K Prefill Trace Test"
echo "=========================================="

run_test "mul_prefill_128k_trace" \
    "${BASE_PATH}/mlp/test_ds_mul.py" \
    "test_ds_mul" \
    "${LONG_SEQ_FILTER}"

# ============================================
# MLP REDUCE SCATTER TESTS
# ============================================
echo "=========================================="
echo "MLP REDUCE SCATTER - 128K Prefill Trace Test"
echo "=========================================="

run_test "reduce_scatter_prefill_128k_trace" \
    "${BASE_PATH}/mlp/test_ds_reduce_scatter_post_ff2.py" \
    "test_ds_reduce_scatter_post_ff2" \
    "${LONG_SEQ_FILTER}"

# ============================================
# MLP ALL GATHER TESTS
# ============================================
echo "=========================================="
echo "MLP ALL GATHER - 128K Prefill Trace Test"
echo "=========================================="

run_test "all_gather_prefill_128k_trace" \
    "${BASE_PATH}/mlp/test_ds_all_gather_preff1_3.py" \
    "test_ds_all_gather_preff1_3" \
    "${LONG_SEQ_FILTER}"

# ============================================
# SUMMARY
# ============================================
echo "=========================================="
echo "Long Sequence (128K) Test Summary"
echo "=========================================="
echo "Total Passed:     $PASSED"
echo "Total Failed:     $FAILED"
echo "Total OOM:        $OOM"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

# Highlight OOM issues
if [ $OOM -gt 0 ]; then
    echo "=========================================="
    echo "⚠️  OOM DETECTED in $OOM test(s)"
    echo "=========================================="
    echo "Check logs for details on which ops ran out of memory."
    echo ""
fi

# Exit with error code if any tests failed or OOM
if [ $FAILED -gt 0 ] || [ $OOM -gt 0 ]; then
    echo "CI FAILED: $FAILED test(s) failed, $OOM test(s) OOM"
    exit 1
else
    echo "CI PASSED: All long sequence tests passed"
    exit 0
fi
