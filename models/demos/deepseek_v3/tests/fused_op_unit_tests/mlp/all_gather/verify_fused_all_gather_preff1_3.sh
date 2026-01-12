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

mkdir -p logs

echo "=========================================="
echo "Step 5: Verify ttnn function by replacing in module"
echo "=========================================="

# Create a temporary modified version of the MLP module
cp models/demos/deepseek_v3/tt/mlp/mlp.py models/demos/deepseek_v3/tt/mlp/mlp.py.backup

# Modify the prefill forward function
sed -i '/# All gather for efficient matmuls/i\        ####################\n        ### AllGather_preff1/3 ##\n        ####################\n        # Import the fused op for verification\n        from models.demos.deepseek_v3.tests.unit.fused_op_unit_tests.mlp.test_ds_fused_all_gather_preff1_3 import ds_fused_all_gather_preff1_3_ttnn\n' models/demos/deepseek_v3/tt/mlp/mlp.py
sed -i 's/x = ttnn.experimental.all_gather_async(x, \*\*ccl.populate_all_gather_runtime_args(cfg\["all_gather"\]))/x = ds_fused_all_gather_preff1_3_ttnn(x, cfg, ccl)/g' models/demos/deepseek_v3/tt/mlp/mlp.py

echo "Running module test with fused op function (decode)..."
timeout 900 pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0] 2>&1 | tee logs/ds_mlp_with_fused_op_decode_$(date +%Y%m%d_%H%M%S).log

echo "Running module test with fused op function (prefill)..."
timeout 900 pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[prefill-512-MLP-None-device_params0] 2>&1 | tee logs/ds_mlp_with_fused_op_prefill_$(date +%Y%m%d_%H%M%S).log

# Restore the original MLP module
mv models/demos/deepseek_v3/tt/mlp/mlp.py.backup models/demos/deepseek_v3/tt/mlp/mlp.py

echo "Module test with fused op function completed successfully!"

echo ""
echo "=========================================="
echo "Step 7: Verify reference and test code with unit test"
echo "=========================================="

echo "Running fused op unit test (decode)..."
timeout 900 pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and eager" 2>&1 | tee logs/ds_fused_all_gather_preff1_3_decode_$(date +%Y%m%d_%H%M%S).log

echo "Running fused op unit test (prefill 128)..."
timeout 900 pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and eager" 2>&1 | tee logs/ds_fused_all_gather_preff1_3_prefill_128_$(date +%Y%m%d_%H%M%S).log

echo "Fused op unit tests completed successfully!"

echo ""
echo "=========================================="
echo "Step 8: Verify configurations match between test and module"
echo "=========================================="

echo "Running device perf test for fused op (decode)..."
export DS_FUSED_ALL_GATHER_PREFF1_3_DEVICE_PERF=1
timeout 900 pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and trace" 2>&1 | tee logs/ds_fused_all_gather_preff1_3_device_perf_decode_$(date +%Y%m%d_%H%M%S).log
unset DS_FUSED_ALL_GATHER_PREFF1_3_DEVICE_PERF

echo "Running device perf test for fused op (prefill)..."
export DS_FUSED_ALL_GATHER_PREFF1_3_DEVICE_PERF=1
timeout 900 pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and eager" 2>&1 | tee logs/ds_fused_all_gather_preff1_3_device_perf_prefill_$(date +%Y%m%d_%H%M%S).log
unset DS_FUSED_ALL_GATHER_PREFF1_3_DEVICE_PERF

echo "Running module test with 1 iteration (decode)..."
timeout 900 pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0] 2>&1 | tee logs/ds_mlp_module_device_perf_decode_$(date +%Y%m%d_%H%M%S).log

echo "Running module test with 1 iteration (prefill)..."
timeout 900 pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[prefill-512-MLP-None-device_params0] 2>&1 | tee logs/ds_mlp_module_device_perf_prefill_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "All verification steps completed!"
echo "=========================================="
echo "Please review the log files in the logs/ directory for detailed results."
