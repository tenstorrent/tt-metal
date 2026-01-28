#!/bin/bash
# Script to verify the AllGather_preff1/3 fused op unit test
# This script should be run from inside the docker container with the python environment activated

set -e

export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
export MESH_DEVICE=TG
export TT_METAL_RUNTIME_ROOT=/home/models-team/hzhou/tt-metal

LOG_DIR="logs/all_gather_fused_op"
mkdir -p $LOG_DIR

echo "=========================================="
echo "AllGather_preff1/3 Fused Op Unit Test Verification"
echo "=========================================="
echo "Test supports: decode/prefill, trace/eager, program_cache on/off"
echo ""

echo "=========================================="
echo "Step 1: Verify reference and test code with unit test"
echo "=========================================="

echo "Running fused op unit test (decode)..."
timeout 900 pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/all_gather/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and not no_program_cache and eager" 2>&1 | tee $LOG_DIR/ds_fused_all_gather_preff1_3_decode_$(date +%Y%m%d_%H%M%S).log

echo "Running fused op unit test (prefill 128)..."
timeout 900 pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/all_gather/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and not no_program_cache and eager" 2>&1 | tee $LOG_DIR/ds_fused_all_gather_preff1_3_prefill_128_$(date +%Y%m%d_%H%M%S).log

echo "Fused op unit tests completed successfully!"

echo ""
echo "=========================================="
echo "Step 2: Run trace mode and device perf tests"
echo "=========================================="

echo "Running trace mode test for fused op (decode)..."
timeout 900 pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/all_gather/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and not no_program_cache and trace" 2>&1 | tee $LOG_DIR/ds_fused_all_gather_preff1_3_trace_decode_$(date +%Y%m%d_%H%M%S).log

echo "Running trace mode test for fused op (prefill)..."
timeout 900 pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/all_gather/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and not no_program_cache and trace" 2>&1 | tee $LOG_DIR/ds_fused_all_gather_preff1_3_trace_prefill_$(date +%Y%m%d_%H%M%S).log

echo "Trace mode tests completed successfully!"

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
