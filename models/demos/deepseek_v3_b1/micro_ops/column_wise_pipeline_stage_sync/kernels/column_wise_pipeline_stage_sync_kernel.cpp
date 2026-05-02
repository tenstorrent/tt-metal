// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified ColumnWisePipelineStageSync Kernel.
 *
 * Uses the unified ColumnWisePipelineStageSync op from unified_kernels/column_wise_pipeline_stage_sync.hpp
 */

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/column_wise_pipeline_stage_sync.hpp"

void kernel_main() {
    using ColumnWisePipelineStageSync = deepseek_b1_ops::ColumnWisePipelineStageSync;

#if defined(COMPILE_FOR_NCRISC)
    // Reader CTArgs
    using CTArgs = ColumnWisePipelineStageSync::ReaderCTArgs<
        get_named_compile_time_arg_val("run_entry_device_logic_on_ncrisc"),
        get_named_compile_time_arg_val("run_exit_device_logic_on_ncrisc"),
        get_named_compile_time_arg_val("entry_device_core_noc_x_addr"),
        get_named_compile_time_arg_val("entry_device_core_noc_y_addr"),
        get_named_compile_time_arg_val("r1_semaphore_l1_addr"),
        get_named_compile_time_arg_val("r2_semaphore_l1_addr"),
        get_named_compile_time_arg_val("r3_semaphore_l1_addr"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Reader runtime args
    ColumnWisePipelineStageSync::ReaderArgs rt_args = {};

#elif defined(COMPILE_FOR_BRISC)
    // Writer CTArgs
    using CTArgs = ColumnWisePipelineStageSync::WriterCTArgs<
        get_named_compile_time_arg_val("run_entry_device_logic_on_brisc"),
        get_named_compile_time_arg_val("run_exit_device_logic_on_brisc"),
        get_named_compile_time_arg_val("entry_device_core_noc_x_addr"),
        get_named_compile_time_arg_val("entry_device_core_noc_y_addr"),
        get_named_compile_time_arg_val("r1_semaphore_l1_addr"),
        get_named_compile_time_arg_val("r2_semaphore_l1_addr"),
        get_named_compile_time_arg_val("r3_semaphore_l1_addr"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Writer runtime args
    ColumnWisePipelineStageSync::WriterArgs rt_args = {};

#elif defined(COMPILE_FOR_TRISC)
    // Compute CTArgs
    using CTArgs = ColumnWisePipelineStageSync::ComputeCTArgs;

    // Compute runtime args
    ColumnWisePipelineStageSync::ComputeArgs rt_args = {};
#endif

    // Execute the op (looped for testing iteration correctness)
    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("num_iterations");
    ColumnWisePipelineStageSync::Op<CTArgs> op;
    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
        op(rt_args);
    }
}
