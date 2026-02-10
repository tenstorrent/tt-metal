#!/bin/bash

# Run CI op tests locally to pre-populate cache and optionally collect device perf
# This matches the CI environment from galaxy-deepseek-tests-impl.yaml
#
# Usage:
#   ./run_ci_op_tests_locally.sh                    # Run functional tests only (cache generation)
#   ./run_ci_op_tests_locally.sh --with-device-perf # Run functional + device perf tests
#   ./run_ci_op_tests_locally.sh --device-perf-only # Run device perf tests only

set -uo pipefail

cd /home/models-team/hzhou/tt-metal
source python_env/bin/activate

# Parse arguments
RUN_FUNCTIONAL=true
RUN_DEVICE_PERF=false

if [ "${1:-}" = "--with-device-perf" ]; then
    RUN_FUNCTIONAL=true
    RUN_DEVICE_PERF=true
elif [ "${1:-}" = "--device-perf-only" ]; then
    RUN_FUNCTIONAL=false
    RUN_DEVICE_PERF=true
fi

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
echo "  Run Functional: $RUN_FUNCTIONAL"
echo "  Run Device Perf: $RUN_DEVICE_PERF"
echo "=============================================="

# Install dependencies
echo "Installing dependencies..."
uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt

# Validate weight cache
echo "Validating weight cache..."
python3 models/demos/deepseek_v3/scripts/validate_weight_cache.py --root "$DEEPSEEK_V3_CACHE/tests_cache" || true

# ============================================
# DeepSeek Op Unit Tests - Functional (Cache Generation)
# ============================================
if [ "$RUN_FUNCTIONAL" = true ]; then
    echo ""
    echo "=============================================="
    echo "Starting Functional Op Tests"
    echo "=============================================="

echo "=============================================="
echo "EMBEDDING op tests"
echo "=============================================="
echo "Running EMBEDDING (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running EMBEDDING (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "EMBEDDING ALL GATHER op tests"
echo "=============================================="
echo "Running EMBEDDING ALL GATHER (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding -k "decode and 1 and trace and program_cache and not no_program_cache" --timeout 600 --durations=0
echo "Running EMBEDDING ALL GATHER (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding -k "prefill and 128 and eager and program_cache and not no_program_cache" --timeout 600 --durations=0

echo "=============================================="
echo "LM HEAD op tests"
echo "=============================================="
echo "Running LM HEAD (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running LM HEAD (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "RMS NORM op tests"
echo "=============================================="
echo "Running RMS NORM (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm -k "decode and 1 and kv_lora_rank_512 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running RMS NORM (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm -k "prefill and 128 and kv_lora_rank_512 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "DISTRIBUTED NORM op tests"
echo "=============================================="
echo "Running DISTRIBUTED NORM (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running DISTRIBUTED NORM (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "MLP FF1/3 op tests"
echo "=============================================="
echo "Running MLP FF1/3 (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3 -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running MLP FF1/3 (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3 -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "MLP FF2 op tests"
echo "=============================================="
echo "Running MLP FF2 (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2 -k "decode and 1 and trace and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0
echo "Running MLP FF2 (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2 -k "prefill and 128 and eager and program_cache and not no_program_cache and real_weights" --timeout 600 --durations=0

echo "=============================================="
echo "MLP MUL op tests"
echo "=============================================="
echo "Running MLP MUL (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul -k "decode and 1 and trace and program_cache and not no_program_cache" --timeout 600 --durations=0
echo "Running MLP MUL (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul -k "prefill and 128 and eager and program_cache and not no_program_cache" --timeout 600 --durations=0

echo "=============================================="
echo "MLP REDUCE SCATTER op tests"
echo "=============================================="
echo "Running MLP REDUCE SCATTER (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2 -k "decode and 1 and trace and program_cache and not no_program_cache" --timeout 600 --durations=0
echo "Running MLP REDUCE SCATTER (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2 -k "prefill and 128 and eager and program_cache and not no_program_cache" --timeout 600 --durations=0

echo "=============================================="
echo "MLP ALL GATHER op tests"
echo "=============================================="
echo "Running MLP ALL GATHER (decode seq_len=1, batch=32, trace)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3 -k "decode and 1 and trace and program_cache and not no_program_cache" --timeout 600 --durations=0
echo "Running MLP ALL GATHER (prefill seq_len=128, batch=32, eager)..."
pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3 -k "prefill and 128 and eager and program_cache and not no_program_cache" --timeout 600 --durations=0

    echo "=============================================="
    echo "All functional op tests completed!"
    echo "=============================================="
else
    echo "Skipping functional tests (cache generation disabled)"
fi

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
    echo "EMBEDDING device perf"
    echo "=============================================="
    echo "Running EMBEDDING device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running EMBEDDING device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "ALL GATHER EMBEDDING device perf"
    echo "=============================================="
    echo "Running ALL GATHER EMBEDDING device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running ALL GATHER EMBEDDING device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "LM HEAD device perf"
    echo "=============================================="
    echo "Running LM HEAD device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running LM HEAD device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "RMS NORM device perf"
    echo "=============================================="
    echo "Running RMS NORM device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running RMS NORM device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "DISTRIBUTED NORM device perf"
    echo "=============================================="
    echo "Running DISTRIBUTED NORM device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running DISTRIBUTED NORM device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "FF1_3 device perf"
    echo "=============================================="
    echo "Running FF1_3 device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running FF1_3 device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "ALL GATHER PREFF1_3 device perf"
    echo "=============================================="
    echo "Running ALL GATHER PREFF1_3 device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running ALL GATHER PREFF1_3 device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "MUL device perf"
    echo "=============================================="
    echo "Running MUL device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running MUL device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "FF2 device perf"
    echo "=============================================="
    echo "Running FF2 device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running FF2 device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "REDUCE SCATTER device perf"
    echo "=============================================="
    echo "Running REDUCE SCATTER device perf (decode seq_len=1, trace + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2_device_perf -k "decode and 1" --timeout 600 --durations=0
    echo "Running REDUCE SCATTER device perf (prefill seq_len=128, eager + program_cache)..."
    pytest models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2_device_perf -k "prefill and 128" --timeout 600 --durations=0

    echo "=============================================="
    echo "All device perf tests completed!"
    echo "=============================================="
else
    echo "Skipping device perf tests (use --with-device-perf to enable)"
fi

echo ""
echo "=============================================="
echo "Test Execution Complete!"
echo "=============================================="
echo "Cache location: $DEEPSEEK_V3_CACHE"
echo ""
if [ "$RUN_FUNCTIONAL" = true ]; then
    echo "✓ Functional tests completed - cache populated:"
    echo "  - Weight cache (converted model weights)"
    echo "  - Program cache (compiled kernels)"
fi
if [ "$RUN_DEVICE_PERF" = true ]; then
    echo "✓ Device perf tests completed:"
    echo "  - Kernel duration metrics collected"
    echo "  - Op-to-op latency metrics collected"
    echo "  - Benchmark data saved to: generated/benchmark_data/"
fi
echo ""
echo "You can now run CI tests using this cache."
echo "=============================================="
