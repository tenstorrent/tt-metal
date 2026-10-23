#!/bin/bash

# Run DeepSeek fused-op device perf cases using direct Tracy commands.
# Intended for CI and local debug parity.

set -euo pipefail

PROFILER_DIR="${PROFILER_DIR:-generated/profiler/deepseek_v3_fused_ops_device_perf}"
TRACY_TIMEOUT_PORT="${TRACY_TIMEOUT_PORT:-5000}"
TRACY_OP_SUPPORT_COUNT="${TRACY_OP_SUPPORT_COUNT:-10000}"

run_tracy_case() {
    local name="$1"
    local perf_env_var="$2"
    local test_target="$3"
    local k_expr="$4"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Case: $name"
    echo "Target: $test_target"
    echo "Expr: $k_expr"
    echo "--------------------------------------------------------------------------------"

    env "${perf_env_var}=1" python3 -m tracy -p -r \
        -o "$PROFILER_DIR" \
        --check-exit-code \
        --op-support-count "$TRACY_OP_SUPPORT_COUNT" \
        -t "$TRACY_TIMEOUT_PORT" \
        -a device_kernel_duration \
        -m "pytest $test_target -k \"$k_expr\""
}

echo "=============================================="
echo "DeepSeek fused-op device perf (direct Tracy)"
echo "=============================================="
echo "Profiler dir: $PROFILER_DIR"
echo "Cases: decode+1(trace+pcache), prefill+128(eager+pcache)"
echo "=============================================="

# Embedding
run_tracy_case \
    "embedding decode seq1 (trace+pcache)" \
    "DS_EMBEDDING_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding" \
    "program_cache and not no_program_cache and trace and decode and 1 and real_weights"
run_tracy_case \
    "embedding prefill seq128 (eager+pcache)" \
    "DS_EMBEDDING_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_embedding.py::test_ds_embedding" \
    "program_cache and not no_program_cache and eager and prefill and 128 and real_weights"

# All-gather embedding
run_tracy_case \
    "all_gather_embedding decode seq1 (trace+pcache)" \
    "DS_ALL_GATHER_EMBEDDING_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "all_gather_embedding prefill seq128 (eager+pcache)" \
    "DS_ALL_GATHER_EMBEDDING_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/embedding/test_ds_all_gather_embedding.py::test_ds_all_gather_embedding" \
    "program_cache and not no_program_cache and eager and prefill and 128"

# LM head
run_tracy_case \
    "lm_head decode seq1 (trace+pcache)" \
    "DS_LM_HEAD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head" \
    "program_cache and not no_program_cache and trace and decode and seq1 and real_weights"
run_tracy_case \
    "lm_head prefill seq128 (eager+pcache)" \
    "DS_LM_HEAD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/lm_head/test_ds_lm_head.py::test_ds_lm_head" \
    "program_cache and not no_program_cache and eager and prefill and seq128 and real_weights"

# RMS norm
run_tracy_case \
    "rms_norm decode seq1 (trace+pcache)" \
    "DS_RMS_NORM_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm" \
    "program_cache and not no_program_cache and trace and decode and 1 and real_weights"
run_tracy_case \
    "rms_norm prefill seq128 (eager+pcache)" \
    "DS_RMS_NORM_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_rms_norm.py::test_ds_rms_norm" \
    "program_cache and not no_program_cache and eager and prefill and 128 and real_weights"

# Distributed norm
run_tracy_case \
    "distributed_norm decode seq1 (trace+pcache)" \
    "DS_DISTRIBUTED_NORM_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm" \
    "program_cache and not no_program_cache and trace and decode and 1 and real_weights"
run_tracy_case \
    "distributed_norm prefill seq128 (eager+pcache)" \
    "DS_DISTRIBUTED_NORM_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/rms_norm/test_ds_distributed_norm.py::test_ds_distributed_norm" \
    "program_cache and not no_program_cache and eager and prefill and 128 and real_weights"

# MLP fused ops
run_tracy_case \
    "ff1_3 decode seq1 (trace+pcache)" \
    "DS_FF1_3_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "ff1_3 prefill seq128 (eager+pcache)" \
    "DS_FF1_3_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff1_3.py::test_ds_ff1_3" \
    "program_cache and not no_program_cache and eager and prefill and 128"

run_tracy_case \
    "ff2 decode seq1 (trace+pcache)" \
    "DS_FF2_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "ff2 prefill seq128 (eager+pcache)" \
    "DS_FF2_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_ff2.py::test_ds_ff2" \
    "program_cache and not no_program_cache and eager and prefill and 128"

run_tracy_case \
    "mul decode seq1 (trace+pcache)" \
    "DS_MUL_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "mul prefill seq128 (eager+pcache)" \
    "DS_MUL_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_mul.py::test_ds_mul" \
    "program_cache and not no_program_cache and eager and prefill and 128"

run_tracy_case \
    "reduce_scatter_post_ff2 decode seq1 (trace+pcache)" \
    "DS_REDUCE_SCATTER_POST_FF2_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "reduce_scatter_post_ff2 prefill seq128 (eager+pcache)" \
    "DS_REDUCE_SCATTER_POST_FF2_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_reduce_scatter_post_ff2.py::test_ds_reduce_scatter_post_ff2" \
    "program_cache and not no_program_cache and eager and prefill and 128"

run_tracy_case \
    "all_gather_preff1_3 decode seq1 (trace+pcache)" \
    "DS_ALL_GATHER_PREFF1_3_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3" \
    "program_cache and not no_program_cache and trace and decode and 1"
run_tracy_case \
    "all_gather_preff1_3 prefill seq128 (eager+pcache)" \
    "DS_ALL_GATHER_PREFF1_3_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mlp/test_ds_all_gather_preff1_3.py::test_ds_all_gather_preff1_3" \
    "program_cache and not no_program_cache and eager and prefill and 128"

# MoE fused op
run_tracy_case \
    "moe decode seq1 (trace+pcache)" \
    "DS_MOE_FORWARD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe.py::test_ds_moe_forward" \
    "test_ds_moe_forward and decode and 1 and real_weights and trace and program_cache and not no_program_cache"
run_tracy_case \
    "moe prefill seq128 (eager+pcache)" \
    "DS_MOE_FORWARD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/test_ds_moe.py::test_ds_moe_forward" \
    "test_ds_moe_forward and prefill and 128 and real_weights and eager and program_cache and not no_program_cache"

# MLA fused op
run_tracy_case \
    "mla decode seq1 (trace+pcache)" \
    "DS_MLA_FORWARD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla.py::test_ds_mla_forward" \
    "test_ds_mla_forward and decode and 1 and real_weights and trace and program_cache and not no_program_cache"
run_tracy_case \
    "mla prefill seq128 (eager+pcache)" \
    "DS_MLA_FORWARD_DEVICE_PERF" \
    "models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_ds_mla.py::test_ds_mla_forward" \
    "test_ds_mla_forward and prefill and 128 and real_weights and eager and program_cache and not no_program_cache"

echo ""
echo "=============================================="
echo "All direct Tracy fused-op perf cases completed"
echo "=============================================="
