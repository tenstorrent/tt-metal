#!/bin/bash
# Verification script for ds_fused_mul fused op unit test

set -e

# Setup environment
cd /home/models-team/hzhou/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
export MESH_DEVICE=TG
export TT_METAL_RUNTIME_ROOT=/home/models-team/hzhou/tt-metal

# Create logs directory
mkdir -p logs/fused_op_mul

LOG_DIR="logs/fused_op_mul"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================="
echo "Step 1: Modifying MLP to use ds_fused_mul_ttnn"
echo "========================================="
# Backup original file
cp models/demos/deepseek_v3/tt/mlp/mlp.py models/demos/deepseek_v3/tt/mlp/mlp.py.bak

# Add import
sed -i '/^from models.demos.deepseek_v3.utils.run_config import/,/)$/s/)$/)\nfrom models.demos.deepseek_v3.tests.fused_op_unit_tests.mlp.mul.test_ds_fused_mul import ds_fused_mul_ttnn/' models/demos/deepseek_v3/tt/mlp/mlp.py

# Replace ttnn.mul calls with ds_fused_mul_ttnn in decode
sed -i 's/activated = ttnn.mul(w1_out_activated, w3_out, \*\*cfg\["mul"\])/activated = ds_fused_mul_ttnn(w1_out_activated, w3_out, cfg)/' models/demos/deepseek_v3/tt/mlp/mlp.py

# Replace ttnn.mul calls with ds_fused_mul_ttnn in prefill
sed -i 's/activated = ttnn.mul(w1_out, w3_out, \*\*cfg\["mul"\])/activated = ds_fused_mul_ttnn(w1_out, w3_out, cfg)/' models/demos/deepseek_v3/tt/mlp/mlp.py

echo "MLP modified to use ds_fused_mul_ttnn"
echo ""

echo "========================================="
echo "Step 2: Running MLP test with ds_fused_mul_ttnn"
echo "========================================="
timeout 600 pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0] -v 2>&1 | tee "${LOG_DIR}/step3_verify_ds_fused_mul_${TIMESTAMP}.log"
echo ""
echo "Verification test completed. Log saved to ${LOG_DIR}/step3_verify_ds_fused_mul_${TIMESTAMP}.log"
echo ""

echo "========================================="
echo "Step 3: Reverting MLP changes"
echo "========================================="
mv models/demos/deepseek_v3/tt/mlp/mlp.py.bak models/demos/deepseek_v3/tt/mlp/mlp.py
echo "MLP reverted to original state"
echo ""

echo "========================================="
echo "Step 4: Running fused op unit test"
echo "========================================="
timeout 600 pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/mul/test_ds_fused_mul.py::test_ds_fused_mul -k "decode and 1 and eager and program_cache" -v 2>&1 | tee "${LOG_DIR}/step5_fused_op_test_${TIMESTAMP}.log"
echo ""
echo "Fused op unit test completed. Log saved to ${LOG_DIR}/step5_fused_op_test_${TIMESTAMP}.log"
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
echo "To verify results:"
echo "1. Check step1 log for baseline PCC (should pass)"
echo "2. Check step3 log - PCC should match baseline"
echo "3. Check step5 log - Fused op unit test PCC should be > 0.99"
echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
