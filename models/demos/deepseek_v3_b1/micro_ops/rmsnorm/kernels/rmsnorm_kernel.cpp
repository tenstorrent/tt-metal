// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// RMSNorm unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout

#include "kernel_op_api.hpp"
#include "rmsnorm.hpp"

KERNEL_ENTRY {
    using RMSNorm = deepseek_b1_ops::RMSNorm;

// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// Compile-time args: [input_cb, scalars_cb, gamma_cb, num_tiles, tiny_tile]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = RMSNorm::ReaderCTArgs<
        get_compile_time_arg_val(0),  // input_cb
        get_compile_time_arg_val(1),  // scalars_cb
        get_compile_time_arg_val(2),  // gamma_cb
        get_compile_time_arg_val(3),  // num_tiles
        get_compile_time_arg_val(4)   // tiny_tile
        >;
// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [output_cb, num_tiles]
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = RMSNorm::WriterCTArgs<
        get_compile_time_arg_val(0),  // output_cb
        get_compile_time_arg_val(1)   // num_tiles
        >;
// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb, fp32_acc, num_tiles, epsilon_index,
// scalar_index]
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = RMSNorm::ComputeCTArgs<
        get_compile_time_arg_val(0),  // input_cb
        get_compile_time_arg_val(1),  // scalars_cb
        get_compile_time_arg_val(2),  // interm_cb
        get_compile_time_arg_val(3),  // gamma_cb
        get_compile_time_arg_val(4),  // output_cb
        get_compile_time_arg_val(5),  // fp32_acc
        get_compile_time_arg_val(6),  // num_tiles
        get_compile_time_arg_val(7),  // epsilon_index
        get_compile_time_arg_val(8),  // scalar_index
        true                          // pop_input
        >;
#endif

    // Instantiate op and build RTArgs based on RISC type
    RMSNorm::Op<CTArgs> rmsnorm;
    using RTArgs = typename RMSNorm::Op<CTArgs>::RTArgs;

#if defined(COMPILE_FOR_NCRISC)
    // Reader has runtime args: epsilon and scalar
    uint32_t arg_idx = 0;
    uint32_t epsilon = get_arg_val<uint32_t>(arg_idx++);
    uint32_t scalar = get_arg_val<uint32_t>(arg_idx++);
    RTArgs rt_args{epsilon, scalar};
#else
    RTArgs rt_args{};
#endif

    rmsnorm(rt_args);
}
KERNEL_END
