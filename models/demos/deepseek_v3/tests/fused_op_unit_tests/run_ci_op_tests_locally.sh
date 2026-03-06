#!/bin/bash

# Run fused-op device perf tests locally (CI-aligned scenarios)
#
# Usage:
#   ./run_ci_op_tests_locally.sh

set -euo pipefail

cd /home/models-team/hzhou/tt-metal
source python_env/bin/activate

RUN_DEVICE_PERF=true

# Set environment variables to match CI
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

echo "=============================================="
echo "DeepSeek V3 Fused Op Tests - Local Execution"
echo "=============================================="
echo "Environment Setup:"
echo "  DEEPSEEK_V3_HF_MODEL: $DEEPSEEK_V3_HF_MODEL"
echo "  DEEPSEEK_V3_CACHE: $DEEPSEEK_V3_CACHE"
echo "  MESH_DEVICE: $MESH_DEVICE"
echo "  Run Device Perf: $RUN_DEVICE_PERF"
echo "=============================================="

# ============================================
# DeepSeek Op Unit Tests - Device Performance
# ============================================
if [ "$RUN_DEVICE_PERF" = true ]; then
    echo ""
    echo "=============================================="
    echo "Starting Device Performance Tests"
    echo "=============================================="
    echo "Collecting kernel and op-to-op latency metrics"
    echo "Configuration: prefill 128 (eager+pcache) + decode 1 (trace+pcache)"
    echo ""

    echo "=============================================="
    echo "DeepSeek fused-op device perf tests (CI-style selection)"
    echo "=============================================="
    CI_FUSED_OP_DEVICE_PERF_K_EXPR='
    (
      (test_ds_embedding_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_all_gather_embedding_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_lm_head_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_rms_norm_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_distributed_norm_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_ff1_3_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_ff2_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_mul_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_reduce_scatter_post_ff2_device_perf and ((decode and 1) or (prefill and 128))) or
      (test_ds_all_gather_preff1_3_device_perf and ((decode and 1) or (prefill and 128)))
    )'
    # Keep expression formatting readable in source but parser-safe for pytest -k.
    CI_FUSED_OP_DEVICE_PERF_K_EXPR="$(printf '%s' "$CI_FUSED_OP_DEVICE_PERF_K_EXPR" | tr '\n' ' ' | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')"
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests -k "$CI_FUSED_OP_DEVICE_PERF_K_EXPR" --timeout 600 --durations=0

    echo "=============================================="
    echo "All device perf tests completed!"
    echo "=============================================="
fi

echo ""
echo "=============================================="
echo "Test Execution Complete!"
echo "=============================================="
if [ "$RUN_DEVICE_PERF" = true ]; then
    echo "✓ Device perf tests completed:"
    echo "  - Kernel duration metrics collected"
    echo "  - Op-to-op latency metrics collected"
    echo "  - Benchmark data saved to: generated/benchmark_data/"
fi
echo ""
echo "=============================================="
