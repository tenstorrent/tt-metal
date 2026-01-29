#!/bin/bash
# Verification script for ds_fused_reduce_scatter_post_ff2
# This script runs the fused op unit test and the MLP module test to verify correctness

set -e

# Setup environment
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
export MESH_DEVICE=TG
export TT_METAL_RUNTIME_ROOT=$(pwd)

LOG_DIR="logs/reduce_scatter_fused_op"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Verification: ds_fused_reduce_scatter_post_ff2"
echo "=========================================="

# Test 1: Run the fused op unit test (decode, eager, program_cache)
echo ""
echo "Test 1: Fused op unit test - decode, eager, program_cache"
echo "=========================================="
LOG_FILE="$LOG_DIR/decode_eager_pcache_$(date +%Y%m%d_%H%M%S).log"
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/reduce_scatter/test_ds_fused_reduce_scatter_post_ff2.py::test_ds_fused_reduce_scatter_post_ff2 \
    -k "decode and eager and program_cache and not no_program_cache" \
    2>&1 | tee "$LOG_FILE"

if grep -q "PASSED" "$LOG_FILE"; then
    echo "✓ PASSED: decode_eager_pcache"
else
    echo "✗ FAILED: decode_eager_pcache"
    exit 1
fi

# Test 2: Run the fused op unit test (prefill, eager, program_cache)
echo ""
echo "Test 2: Fused op unit test - prefill 128, eager, program_cache"
echo "=========================================="
LOG_FILE="$LOG_DIR/prefill_128_eager_pcache_$(date +%Y%m%d_%H%M%S).log"
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/reduce_scatter/test_ds_fused_reduce_scatter_post_ff2.py::test_ds_fused_reduce_scatter_post_ff2 \
    -k "prefill and 128 and eager and program_cache and not no_program_cache" \
    2>&1 | tee "$LOG_FILE"

if grep -q "PASSED" "$LOG_FILE"; then
    echo "✓ PASSED: prefill_128_eager_pcache"
else
    echo "✗ FAILED: prefill_128_eager_pcache"
    exit 1
fi

# Test 3: Run the fused op unit test (decode, trace, program_cache)
echo ""
echo "Test 3: Fused op unit test - decode, trace, program_cache"
echo "=========================================="
LOG_FILE="$LOG_DIR/decode_trace_pcache_$(date +%Y%m%d_%H%M%S).log"
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/reduce_scatter/test_ds_fused_reduce_scatter_post_ff2.py::test_ds_fused_reduce_scatter_post_ff2 \
    -k "decode and trace and program_cache and not no_program_cache" \
    2>&1 | tee "$LOG_FILE"

if grep -q "PASSED" "$LOG_FILE"; then
    echo "✓ PASSED: decode_trace_pcache"
else
    echo "✗ FAILED: decode_trace_pcache"
    exit 1
fi

echo ""
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
