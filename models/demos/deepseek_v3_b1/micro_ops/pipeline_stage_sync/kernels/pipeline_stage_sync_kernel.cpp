// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
        get_named_compile_time_arg_val("is_stalling_device_equal_signalling_device"),
        get_named_compile_time_arg_val("stalling_device_chip_id"),
        get_named_compile_time_arg_val("stalling_device_mesh_id"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Reader runtime args (from common args)
    PipelineStageSync::ReaderArgs rt_args = {
        get_common_arg_val<uint32_t>(0),  // stalling_device_semaphore_noc_x_addr
        get_common_arg_val<uint32_t>(1),  // stalling_device_semaphore_noc_y_addr
        get_common_arg_val<uint32_t>(2),  // stalling_device_semaphore_l1_addr
    };

#elif defined(COMPILE_FOR_BRISC)
    // Writer CTArgs
    using CTArgs = PipelineStageSync::WriterCTArgs<
        get_named_compile_time_arg_val("run_stalling_logic_on_brisc"),
        get_named_compile_time_arg_val("run_signalling_logic_on_brisc"),
        get_named_compile_time_arg_val("is_stalling_device_equal_signalling_device"),
        get_named_compile_time_arg_val("stalling_device_chip_id"),
        get_named_compile_time_arg_val("stalling_device_mesh_id"),
        get_named_compile_time_arg_val("fabric_arg_base")>;

    // Writer runtime args (from common args)
    PipelineStageSync::WriterArgs rt_args = {
        get_common_arg_val<uint32_t>(0),  // stalling_device_semaphore_noc_x_addr
        get_common_arg_val<uint32_t>(1),  // stalling_device_semaphore_noc_y_addr
        get_common_arg_val<uint32_t>(2),  // stalling_device_semaphore_l1_addr
    };

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
