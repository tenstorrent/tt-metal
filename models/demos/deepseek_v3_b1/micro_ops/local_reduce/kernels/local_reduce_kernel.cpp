// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Local Reduce unified kernel
// Element-wise sum reduction: output = in_cb[0] + in_cb[1] + ... + in_cb[n-1]
// Optionally applies SiLU activation: output = SiLU(sum)
//
// NCRISC: Signals sharded CB is ready
// BRISC: No-op
// TRISC: Performs reduction via LocalReduce::Op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/local_reduce.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using LocalReduceCTArgs = deepseek_b1_ops::LocalReduce::ReaderCTArgs;

    constexpr uint32_t in_cb = get_named_compile_time_arg_val("local_reduce_in_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("local_reduce_num_tiles");

    // Setup sharded buffer
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(in_cb, num_tiles);
    }

    deepseek_b1_ops::LocalReduce::ReaderArgs local_reduce_args{};

#elif defined(COMPILE_FOR_BRISC)
    using LocalReduceCTArgs = deepseek_b1_ops::LocalReduce::WriterCTArgs;

    deepseek_b1_ops::LocalReduce::WriterArgs local_reduce_args{};

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("local_reduce_in_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("local_reduce_out_cb");

    using LocalReduceCTArgs = deepseek_b1_ops::LocalReduce::ComputeCTArgs<
        get_named_compile_time_arg_val("local_reduce_num_tiles"),
        get_named_compile_time_arg_val("local_reduce_apply_silu") == 1>;

    deepseek_b1_ops::LocalReduce::ComputeArgs local_reduce_args{
        .in_cb = in_cb,
        .out_cb = out_cb,
    };
    deepseek_compute_kernel_init();
#endif

    deepseek_b1_ops::LocalReduce::Op<LocalReduceCTArgs, Core::is_active_core> local_reduce;
    local_reduce(local_reduce_args);
}
