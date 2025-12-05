// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: RMSNorm
// - NCRISC: RMSNorm reader
// - BRISC: RMSNorm writer
// - TRISC: RMSNorm compute

#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/kernel_op_api.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/rmsnorm.hpp"
#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/unified_core_descriptor.hpp"

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

    RMSNorm::Op<CTArgs>::ReaderArgs rt_args;
    rt_args.epsilon = get_arg_val<uint32_t>(0);
    rt_args.scalar = get_arg_val<uint32_t>(1);

// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [output_cb, num_tiles]
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = RMSNorm::WriterCTArgs<
        get_compile_time_arg_val(0),  // output_cb
        get_compile_time_arg_val(1)   // num_tiles
        >;

    RMSNorm::Op<CTArgs>::WriterArgs rt_args;

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb,
//                     fp32_acc, num_tiles, epsilon_index, scalar_index]
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

    RMSNorm::Op<CTArgs>::ComputeArgs rt_args;
#endif

    // Use UnifiedCoreDescriptor for compile-time role checks
    using Core = pre_sdpa::UnifiedCoreDescriptor;

    // Only input cores run RMSNorm - dead code elimination for others
    if constexpr (Core::is_input_core) {
        RMSNorm::Op<CTArgs> rmsnorm;
        rmsnorm(rt_args);
    }
}
KERNEL_END
