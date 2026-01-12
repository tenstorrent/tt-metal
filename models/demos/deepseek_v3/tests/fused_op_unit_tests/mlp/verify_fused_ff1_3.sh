#!/bin/bash
# Script to verify the FF1/3 fused op unit test (gate + up projections)
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

TEST_FILE="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_fused_ff1_3.py"
LOG_DIR="logs/ff1_3_fused_op"
TIMEOUT=900  # 15 minutes

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FF1/3 Fused Op Unit Test Verification"
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

    if timeout $TIMEOUT pytest "$TEST_FILE::test_ds_fused_ff1_3" -k "$filter" -v 2>&1 | tee "$log_file"; then
        echo "✓ PASSED: $test_name"
    else
        echo "✗ FAILED: $test_name (see $log_file)"
    fi
    echo ""
}

echo "=========================================="
echo "Step 1: Run baseline MLP module test"
echo "=========================================="
echo "Running MLP module test (decode) to verify baseline..."
timeout $TIMEOUT pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0] -v 2>&1 | tee "$LOG_DIR/mlp_baseline_decode_$(date +%Y%m%d_%H%M%S).log" || true
echo ""

echo "=========================================="
echo "Step 2: Eager mode tests (no trace)"
echo "=========================================="

# Decode - Eager mode with program cache
run_test "decode_eager_pcache_real" "decode and 1 and program_cache and not no_program_cache and eager and real_weights"
run_test "decode_eager_pcache_random" "decode and 1 and program_cache and not no_program_cache and eager and random_weights"

# Prefill 128 - Eager mode with program cache
run_test "prefill_128_eager_pcache_real" "prefill and 128 and program_cache and not no_program_cache and eager and real_weights"
run_test "prefill_128_eager_pcache_random" "prefill and 128 and program_cache and not no_program_cache and eager and random_weights"

# Prefill 1024 - Eager mode with program cache
run_test "prefill_1024_eager_pcache_real" "prefill and 1024 and program_cache and not no_program_cache and eager and real_weights"

# Prefill 8192 - Eager mode with program cache (this tests chunking)
run_test "prefill_8192_eager_pcache_real" "prefill and 8192 and program_cache and not no_program_cache and eager and real_weights"

echo "=========================================="
echo "Step 3: Trace mode tests"
echo "=========================================="

# Decode - Trace mode with program cache
run_test "decode_trace_pcache_real" "decode and 1 and program_cache and not no_program_cache and trace and real_weights"
run_test "decode_trace_pcache_random" "decode and 1 and program_cache and not no_program_cache and trace and random_weights"

# Prefill 128 - Trace mode with program cache
run_test "prefill_128_trace_pcache_real" "prefill and 128 and program_cache and not no_program_cache and trace and real_weights"

# Prefill 1024 - Trace mode with program cache
run_test "prefill_1024_trace_pcache_real" "prefill and 1024 and program_cache and not no_program_cache and trace and real_weights"

# Prefill 8192 - Trace mode with program cache
run_test "prefill_8192_trace_pcache_real" "prefill and 8192 and program_cache and not no_program_cache and trace and real_weights"

echo "=========================================="
echo "Step 4: Program cache disabled tests (eager only)"
echo "=========================================="

# Decode - No program cache
run_test "decode_eager_no_pcache_real" "decode and 1 and no_program_cache and eager and real_weights"

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
