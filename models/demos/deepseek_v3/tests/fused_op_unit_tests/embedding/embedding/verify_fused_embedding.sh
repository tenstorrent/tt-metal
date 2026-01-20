#!/bin/bash
# Script to verify the embedding fused op unit test
# This script should be run from the tt-metal root directory inside the docker container

set -e

# Environment setup
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
export MESH_DEVICE=TG
export TT_METAL_RUNTIME_ROOT=/home/models-team/hzhou/tt-metal

TEST_FILE="models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_fused_embedding.py"
LOG_DIR="logs/embedding_fused_op"
TIMEOUT=900  # 15 minutes

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Embedding Fused Op Unit Test Verification"
echo "=========================================="
echo "Test file: $TEST_FILE"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a test with logging
run_test() {
    local test_name="$1"
    local filter="$2"
    local log_file="$LOG_DIR/${test_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "Filter: $filter"
    echo "Log: $log_file"
    echo "----------------------------------------"

    if timeout $TIMEOUT pytest "$TEST_FILE::test_ds_fused_embedding" -k "$filter" -v 2>&1 | tee "$log_file"; then
        echo "[PASSED] $test_name"
    else
        echo "[FAILED] $test_name (see $log_file)"
    fi
    echo ""
}

echo "=========================================="
echo "Step 1: Eager mode tests (no trace)"
echo "=========================================="

# Decode - Eager mode with program cache
run_test "decode_eager_pcache_real" "decode and 32 and program_cache and not no_program_cache and eager and real_weights"
run_test "decode_eager_pcache_random" "decode and 32 and program_cache and not no_program_cache and eager and random_weights"

# Prefill 128 - Eager mode with program cache
run_test "prefill_128_eager_pcache_real" "prefill and 128 and program_cache and not no_program_cache and eager and real_weights"
run_test "prefill_128_eager_pcache_random" "prefill and 128 and program_cache and not no_program_cache and eager and random_weights"

# Prefill 512 - Eager mode with program cache
run_test "prefill_512_eager_pcache_real" "prefill and 512 and program_cache and not no_program_cache and eager and real_weights"

# Prefill 2048 - Eager mode with program cache
run_test "prefill_2048_eager_pcache_real" "prefill and 2048 and program_cache and not no_program_cache and eager and real_weights"

echo "=========================================="
echo "Step 2: Trace mode tests"
echo "=========================================="

# Decode - Trace mode with program cache
run_test "decode_trace_pcache_real" "decode and 32 and program_cache and not no_program_cache and trace and real_weights"
run_test "decode_trace_pcache_random" "decode and 32 and program_cache and not no_program_cache and trace and random_weights"

# Prefill 128 - Trace mode with program cache
run_test "prefill_128_trace_pcache_real" "prefill and 128 and program_cache and not no_program_cache and trace and real_weights"

# Prefill 512 - Trace mode with program cache
run_test "prefill_512_trace_pcache_real" "prefill and 512 and program_cache and not no_program_cache and trace and real_weights"

# Prefill 2048 - Trace mode with program cache
run_test "prefill_2048_trace_pcache_real" "prefill and 2048 and program_cache and not no_program_cache and trace and real_weights"

echo "=========================================="
echo "Step 3: Program cache disabled tests (eager only)"
echo "=========================================="

# Decode - No program cache
run_test "decode_eager_no_pcache_real" "decode and 32 and no_program_cache and eager and real_weights"

# Prefill 128 - No program cache
run_test "prefill_128_eager_no_pcache_real" "prefill and 128 and no_program_cache and eager and real_weights"

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "All test logs are saved in: $LOG_DIR"
echo ""
echo "To view failed tests:"
echo "  grep -l 'FAILED' $LOG_DIR/*.log"
echo ""
echo "To check PCC values:"
echo "  grep 'PCC:' $LOG_DIR/*.log"
echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
