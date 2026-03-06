#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# CI Test Script for DeepSeek V3 Op Unit Tests

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
LOG_DIR="${TT_METAL_HOME}/logs/ci_ops_$(date +%Y%m%d_%H%M%S)"
TIMEOUT=${TIMEOUT:-1800}  # 30 minutes per test

# Create log directory
mkdir -p "$LOG_DIR"

# Track results
PASSED=0
FAILED=0
SKIPPED=0

echo "=============================================="
echo "DeepSeek V3 Op CI Tests"
echo "=============================================="
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

    if timeout $TIMEOUT python -m pytest "$test_path::$test_func" -k "$test_filter" -v 2>&1 | tee "$log_file"; then
        if grep -q "PASSED" "$log_file"; then
            echo "✓ $test_name: PASSED"
            PASSED=$((PASSED + 1))
        elif grep -q "SKIPPED" "$log_file"; then
            echo "- $test_name: SKIPPED"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "? $test_name: UNKNOWN"
        fi
    else
        if grep -q "SKIPPED" "$log_file"; then
            echo "- $test_name: SKIPPED"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "✗ $test_name: FAILED"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
}

# ============================================
# EMBEDDING TESTS
# ============================================
echo "=========================================="
echo "EMBEDDING Tests"
echo "=========================================="

EMBEDDING_PATH="${BASE_PATH}/embedding/test_ds_embedding.py"

run_test "embedding_decode_seq1_trace" \
    "$EMBEDDING_PATH" \
    "test_ds_embedding" \
    "decode and 1 and trace and program_cache and not no_program_cache"

run_test "embedding_prefill_seq128_eager" \
    "$EMBEDDING_PATH" \
    "test_ds_embedding" \
    "prefill and 128 and eager and program_cache and not no_program_cache"

# ============================================
# EMBEDDING ALL GATHER TESTS
# ============================================
echo "=========================================="
echo "EMBEDDING ALL GATHER Tests"
echo "=========================================="

EMBEDDING_AG_PATH="${BASE_PATH}/embedding/test_ds_all_gather_embedding.py"

run_test "embedding_all_gather_decode_seq1_trace" \
    "$EMBEDDING_AG_PATH" \
    "test_ds_all_gather_embedding" \
    "decode and 1 and trace and program_cache and not no_program_cache"

run_test "embedding_all_gather_prefill_seq128_eager" \
    "$EMBEDDING_AG_PATH" \
    "test_ds_all_gather_embedding" \
    "prefill and 128 and eager and program_cache and not no_program_cache"

# ============================================
# LM HEAD TESTS
# ============================================
echo "=========================================="
echo "LM HEAD Tests"
echo "=========================================="

LM_HEAD_PATH="${BASE_PATH}/lm_head/test_ds_lm_head.py"

run_test "lm_head_decode_seq1_trace" \
    "$LM_HEAD_PATH" \
    "test_ds_lm_head" \
    "decode and 1 and trace and program_cache and not no_program_cache and real_weights"

run_test "lm_head_prefill_seq128_eager" \
    "$LM_HEAD_PATH" \
    "test_ds_lm_head" \
    "prefill and 128 and eager and program_cache and not no_program_cache and real_weights"

# ============================================
# RMS NORM TESTS
# ============================================
echo "=========================================="
echo "RMS NORM Tests"
echo "=========================================="

RMS_NORM_PATH="${BASE_PATH}/rms_norm/test_ds_rms_norm.py"

run_test "rms_norm_decode_seq1_trace" \
    "$RMS_NORM_PATH" \
    "test_ds_rms_norm" \
    "decode and 1 and kv_lora_rank_512 and trace and program_cache and not no_program_cache and real_weights"

run_test "rms_norm_prefill_seq128_eager" \
    "$RMS_NORM_PATH" \
    "test_ds_rms_norm" \
    "prefill and 128 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and real_weights"

# ============================================
# DISTRIBUTED NORM TESTS
# ============================================
echo "=========================================="
echo "DISTRIBUTED NORM Tests"
echo "=========================================="

DIST_NORM_PATH="${BASE_PATH}/rms_norm/test_ds_distributed_norm.py"

run_test "distributed_norm_decode_seq1_trace" \
    "$DIST_NORM_PATH" \
    "test_ds_distributed_norm" \
    "decode and 1 and trace and program_cache and not no_program_cache and real_weights"

run_test "distributed_norm_prefill_seq128_eager" \
    "$DIST_NORM_PATH" \
    "test_ds_distributed_norm" \
    "prefill and 128 and eager and program_cache and not no_program_cache and real_weights"

# ============================================
# MLP FF1/3 TESTS
# ============================================
echo "=========================================="
echo "MLP FF1/3 Tests"
echo "=========================================="

FF1_3_PATH="${BASE_PATH}/mlp/test_ds_ff1_3.py"

run_test "ff1_3_decode_seq1_trace" \
    "$FF1_3_PATH" \
    "test_ds_ff1_3" \
    "decode and 1 and trace and program_cache and not no_program_cache and real_weights"

run_test "ff1_3_prefill_seq128_eager" \
    "$FF1_3_PATH" \
    "test_ds_ff1_3" \
    "prefill and 128 and eager and program_cache and not no_program_cache and real_weights"

# ============================================
# MLP FF2 TESTS
# ============================================
echo "=========================================="
echo "MLP FF2 Tests"
echo "=========================================="

FF2_PATH="${BASE_PATH}/mlp/test_ds_ff2.py"

run_test "ff2_decode_seq1_trace" \
    "$FF2_PATH" \
    "test_ds_ff2" \
    "decode and 1 and trace and program_cache and not no_program_cache and real_weights"

run_test "ff2_prefill_seq128_eager" \
    "$FF2_PATH" \
    "test_ds_ff2" \
    "prefill and 128 and eager and program_cache and not no_program_cache and real_weights"

# ============================================
# MLP MUL TESTS
# ============================================
echo "=========================================="
echo "MLP MUL Tests"
echo "=========================================="

MUL_PATH="${BASE_PATH}/mlp/test_ds_mul.py"

run_test "mul_decode_seq1_trace" \
    "$MUL_PATH" \
    "test_ds_mul" \
    "decode and 1 and trace and program_cache and not no_program_cache"

run_test "mul_prefill_seq128_eager" \
    "$MUL_PATH" \
    "test_ds_mul" \
    "prefill and 128 and eager and program_cache and not no_program_cache"

# ============================================
# MLP REDUCE SCATTER TESTS
# ============================================
echo "=========================================="
echo "MLP REDUCE SCATTER Tests"
echo "=========================================="

RS_PATH="${BASE_PATH}/mlp/test_ds_reduce_scatter_post_ff2.py"

run_test "reduce_scatter_decode_seq1_trace" \
    "$RS_PATH" \
    "test_ds_reduce_scatter_post_ff2" \
    "decode and 1 and trace and program_cache and not no_program_cache"

run_test "reduce_scatter_prefill_seq128_eager" \
    "$RS_PATH" \
    "test_ds_reduce_scatter_post_ff2" \
    "prefill and 128 and eager and program_cache and not no_program_cache"

# ============================================
# MLP ALL GATHER TESTS
# ============================================
echo "=========================================="
echo "MLP ALL GATHER Tests"
echo "=========================================="

AG_PATH="${BASE_PATH}/mlp/test_ds_all_gather_preff1_3.py"

run_test "all_gather_decode_seq1_trace" \
    "$AG_PATH" \
    "test_ds_all_gather_preff1_3" \
    "decode and 1 and trace and program_cache and not no_program_cache"

run_test "all_gather_prefill_seq128_eager" \
    "$AG_PATH" \
    "test_ds_all_gather_preff1_3" \
    "prefill and 128 and eager and program_cache and not no_program_cache"

# ============================================
# SUMMARY
# ============================================
echo "=========================================="
echo "CI Test Summary"
echo "=========================================="
echo "Total Passed:  $PASSED"
echo "Total Failed:  $FAILED"
echo "Total Skipped: $SKIPPED"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

# Exit with error code if any tests failed
if [ $FAILED -gt 0 ]; then
    echo "CI FAILED: $FAILED test(s) failed"
    exit 1
else
    echo "CI PASSED: All tests passed or skipped"
    exit 0
fi
