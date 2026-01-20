#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Verification script for ds_fused_ff2 fused op unit test
# This script runs the test with various configurations to verify correctness
# Supports: decode/prefill modes, trace/eager execution, program_cache on/off

set -e

# Setup environment
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528}
export DEEPSEEK_V3_CACHE=${DEEPSEEK_V3_CACHE:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache}
export MESH_DEVICE=${MESH_DEVICE:-TG}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_HOME}

TEST_PATH="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/ff2/test_ds_fused_ff2.py"
LOG_DIR="logs/ff2_fused_op"
mkdir -p $LOG_DIR

echo "=========================================="
echo "FF2 Fused Op Unit Test Verification"
echo "=========================================="
echo "Test supports: decode/prefill, trace/eager, program_cache on/off"
echo ""

# Test 1: Decode mode, eager execution, with program cache
echo "Test 1: Decode mode, eager, program_cache, real_weights"
echo "--------------------------------------------------------"
pytest ${TEST_PATH}::test_ds_fused_ff2 \
    -k "decode and 1 and eager and program_cache and not no_program_cache and real_weights" \
    -v --timeout=900 2>&1 | tee ${LOG_DIR}/decode_eager_pcache_$(date +%Y%m%d_%H%M%S).log

# Test 2: Decode mode, trace execution, with program cache
echo ""
echo "Test 2: Decode mode, trace, program_cache, real_weights"
echo "--------------------------------------------------------"
pytest ${TEST_PATH}::test_ds_fused_ff2 \
    -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" \
    -v --timeout=900 2>&1 | tee ${LOG_DIR}/decode_trace_pcache_$(date +%Y%m%d_%H%M%S).log

# Test 3: Prefill mode (128 seq_len), eager execution, with program cache
echo ""
echo "Test 3: Prefill mode (seq_len=128), eager, program_cache, real_weights"
echo "----------------------------------------------------------------------"
pytest ${TEST_PATH}::test_ds_fused_ff2 \
    -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" \
    -v --timeout=900 2>&1 | tee ${LOG_DIR}/prefill_128_eager_pcache_$(date +%Y%m%d_%H%M%S).log

# Test 4: Decode mode, eager execution, without program cache
echo ""
echo "Test 4: Decode mode, eager, no_program_cache, real_weights"
echo "----------------------------------------------------------"
pytest ${TEST_PATH}::test_ds_fused_ff2 \
    -k "decode and 1 and eager and no_program_cache and real_weights" \
    -v --timeout=900 2>&1 | tee ${LOG_DIR}/decode_eager_no_pcache_$(date +%Y%m%d_%H%M%S).log

# Test 5: Decode mode, eager, program_cache, random_weights
echo ""
echo "Test 5: Decode mode, eager, program_cache, random_weights"
echo "---------------------------------------------------------"
pytest ${TEST_PATH}::test_ds_fused_ff2 \
    -k "decode and 1 and eager and program_cache and not no_program_cache and random_weights" \
    -v --timeout=900 2>&1 | tee ${LOG_DIR}/decode_eager_pcache_random_$(date +%Y%m%d_%H%M%S).log

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
