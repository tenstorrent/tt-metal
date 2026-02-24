// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Eltwise Add kernel for MoE fused operation
// Adds fused_add tensor (indexed by sender_index) to down_proj output
// Uses CB read pointer update instead of copy for efficiency

#include "../../unified_kernels/eltwise_add.hpp"
#include "../../unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded buffers - signal that input data is ready
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("add_cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("add_cb_in1");
    constexpr uint32_t cb_in0_num_tiles =
        get_named_compile_time_arg_val("add_cb_in0_wait_tiles");  // pages in 32x32-view CB
    constexpr uint32_t cb_in1_wait_tiles =
        get_named_compile_time_arg_val("add_cb_in1_wait_tiles");  // pages in 32x32-view CB

    unified_kernels::setup_sharded_buffer(cb_in0, cb_in0_num_tiles);
    unified_kernels::setup_sharded_buffer(cb_in1, cb_in1_wait_tiles);

    using CTArgs = deepseek_b1_ops::EltwiseAdd::ReaderCTArgs;
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = deepseek_b1_ops::EltwiseAdd::WriterCTArgs;
#elif defined(COMPILE_FOR_TRISC)
    // TRISC: Element-wise add with CB read pointer update for indexed access
    using CTArgs = deepseek_b1_ops::EltwiseAdd::ComputeCTArgs<
        get_named_compile_time_arg_val("add_cb_in0"),
        get_named_compile_time_arg_val("add_cb_in1"),
        get_named_compile_time_arg_val("add_cb_out"),
        get_named_compile_time_arg_val("add_num_tiles"),
        get_named_compile_time_arg_val("add_cb_in0"),  // cb_in0_wait = cb_in0 (same CB)
        get_named_compile_time_arg_val("add_cb_in0_wait_tiles"),
        get_named_compile_time_arg_val("add_cb_in1_wait_tiles"),
        get_named_compile_time_arg_val("add_sender_index"),
        get_named_compile_time_arg_val("add_slice_size_bytes")>;
    deepseek_compute_kernel_init();
#endif

    // Execute
    deepseek_b1_ops::EltwiseAdd::Op<CTArgs, true> add_op;
    add_op();
}
