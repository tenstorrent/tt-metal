// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified PipelineStageSync Kernel.
 *
 * Uses the unified PipelineStageSync op from unified_kernels/pipeline_stage_sync.hpp
 */

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/pipeline_stage_sync.hpp"

void kernel_main() {
    using PipelineStageSync = deepseek_b1_ops::PipelineStageSync;

#if defined(COMPILE_FOR_NCRISC)
    // Reader CTArgs
    using CTArgs = PipelineStageSync::ReaderCTArgs<
        get_named_compile_time_arg_val("run_stalling_logic_on_ncrisc"),
        get_named_compile_time_arg_val("run_signalling_logic_on_ncrisc"),
        get_named_compile_time_arg_val("is_intermediate_signaller"),
        get_named_compile_time_arg_val("stalling_device_semaphore_noc_x_addr"),
        get_named_compile_time_arg_val("stalling_device_semaphore_noc_y_addr"),
        get_named_compile_time_arg_val("stalling_device_semaphore_l1_addr"),
        get_named_compile_time_arg_val("stalling_device_chip_id"),
        get_named_compile_time_arg_val("stalling_device_mesh_id"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Reader runtime args
    PipelineStageSync::ReaderArgs rt_args = {};

#elif defined(COMPILE_FOR_BRISC)
    // Writer CTArgs
    using CTArgs = PipelineStageSync::WriterCTArgs<
        get_named_compile_time_arg_val("run_stalling_logic_on_brisc"),
        get_named_compile_time_arg_val("run_signalling_logic_on_brisc"),
        get_named_compile_time_arg_val("is_intermediate_signaller"),
        get_named_compile_time_arg_val("stalling_device_semaphore_noc_x_addr"),
        get_named_compile_time_arg_val("stalling_device_semaphore_noc_y_addr"),
        get_named_compile_time_arg_val("stalling_device_semaphore_l1_addr"),
        get_named_compile_time_arg_val("stalling_device_chip_id"),
        get_named_compile_time_arg_val("stalling_device_mesh_id"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Writer runtime args
    PipelineStageSync::WriterArgs rt_args = {};

#elif defined(COMPILE_FOR_TRISC)
    // Compute CTArgs
    using CTArgs = PipelineStageSync::ComputeCTArgs;

    // Compute runtime args
    PipelineStageSync::ComputeArgs rt_args = {};
#endif

    // Execute the op (looped for testing iteration correctness)
    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("num_iterations");
    PipelineStageSync::Op<CTArgs> op;
    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
        op(rt_args);
    }
}
