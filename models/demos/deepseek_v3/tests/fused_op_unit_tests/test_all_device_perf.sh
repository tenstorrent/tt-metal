#!/bin/bash

# Test all device perf tests to ensure they work and collect baselines
set -uo pipefail

cd /home/models-team/hzhou/tt-metal
source python_env/bin/activate

# Set environment variables
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$(pwd)/build/lib
export LOGURU_LEVEL=INFO
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI
export MESH_DEVICE=TG
export CI=true
export GITHUB_ACTIONS=true
export TT_METAL_RUNTIME_ROOT=/home/models-team/hzhou/tt-metal

# Output file for collected baselines
BASELINE_FILE="device_perf_baselines.json"
echo "{" > "$BASELINE_FILE"

# Array of all fused ops with their test paths
declare -A OPS_MAP
OPS_MAP["embedding"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding_device_perf"
OPS_MAP["all_gather_embedding"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding_device_perf"
OPS_MAP["lm_head"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head_device_perf"
OPS_MAP["rms_norm"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm_device_perf"
OPS_MAP["distributed_norm"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm_device_perf"
OPS_MAP["ff1_3"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3_device_perf"
OPS_MAP["all_gather_preff1_3"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3_device_perf"
OPS_MAP["mul"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul_device_perf"
OPS_MAP["ff2"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2_device_perf"
OPS_MAP["reduce_scatter"]="models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2_device_perf"

first_op=true

for op_name in "${!OPS_MAP[@]}"; do
    test_path="${OPS_MAP[$op_name]}"

    echo "=============================================="
    echo "Testing: $op_name"
    echo "=============================================="

    if [ "$first_op" = false ]; then
        echo "," >> "$BASELINE_FILE"
    fi
    first_op=false

    echo "  \"$op_name\": {" >> "$BASELINE_FILE"

    # Test prefill seq_len=128 (eager + program_cache)
    echo ""
    echo "Running $op_name prefill seq_len=128 (eager + program_cache)..."
    rm -rf generated/profiler/deepseek_v3_fused_ops_device_perf

    OUTPUT=$(pytest "$test_path" -k "prefill and 128" --timeout 600 -v -s 2>&1)
    PREFILL_KERNEL=$(echo "$OUTPUT" | grep "Device perf totals:" | grep -oP "kernel=\K[0-9.]+")
    PREFILL_OP_TO_OP=$(echo "$OUTPUT" | grep "Device perf totals:" | grep -oP "op_to_op=\K[0-9.]+")

    echo "  Prefill 128: kernel=${PREFILL_KERNEL}us, op_to_op=${PREFILL_OP_TO_OP}us"

    # Test decode seq_len=1 (trace + program_cache)
    echo ""
    echo "Running $op_name decode seq_len=1 (trace + program_cache)..."
    rm -rf generated/profiler/deepseek_v3_fused_ops_device_perf

    OUTPUT=$(pytest "$test_path" -k "decode and 1" --timeout 600 -v -s 2>&1)
    DECODE_KERNEL=$(echo "$OUTPUT" | grep "Device perf totals:" | grep -oP "kernel=\K[0-9.]+")
    DECODE_OP_TO_OP=$(echo "$OUTPUT" | grep "Device perf totals:" | grep -oP "op_to_op=\K[0-9.]+")

    echo "  Decode 1: kernel=${DECODE_KERNEL}us, op_to_op=${DECODE_OP_TO_OP}us"

    # Write to baseline file
    echo "    \"prefill_128\": {" >> "$BASELINE_FILE"
    echo "      \"kernel_us\": $PREFILL_KERNEL," >> "$BASELINE_FILE"
    echo "      \"op_to_op_us\": $PREFILL_OP_TO_OP" >> "$BASELINE_FILE"
    echo "    }," >> "$BASELINE_FILE"
    echo "    \"decode_1\": {" >> "$BASELINE_FILE"
    echo "      \"kernel_us\": $DECODE_KERNEL," >> "$BASELINE_FILE"
    echo "      \"op_to_op_us\": $DECODE_OP_TO_OP" >> "$BASELINE_FILE"
    echo "    }" >> "$BASELINE_FILE"
    echo "  }" >> "$BASELINE_FILE"

    echo ""
done

echo "}" >> "$BASELINE_FILE"

echo "=============================================="
echo "All Device Perf Tests Complete!"
echo "Baselines saved to: $BASELINE_FILE"
echo "=============================================="

cat "$BASELINE_FILE"
