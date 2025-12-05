// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Matmul unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout

#include "kernel_op_api.hpp"
#include "matmul.hpp"

KERNEL_ENTRY {
    using Matmul = deepseek_b1_ops::Matmul;

// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// Compile-time args: [in0_cb, in1_cb, num_tiles_k]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = Matmul::ReaderCTArgs<
        get_compile_time_arg_val(0),  // in0_cb
        get_compile_time_arg_val(1),  // in1_cb
        get_compile_time_arg_val(2),  // num_tiles_k
        true                          // pop_inputs
        >;
// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [out_cb]
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = Matmul::WriterCTArgs<get_compile_time_arg_val(0)  // out_cb
                                        >;
// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = Matmul::ComputeCTArgs<
        get_compile_time_arg_val(0),  // in0_cb
        get_compile_time_arg_val(1),  // in1_cb
        get_compile_time_arg_val(2),  // out_cb
        get_compile_time_arg_val(3),  // interm_cb
        get_compile_time_arg_val(4),  // num_tiles_k
        get_compile_time_arg_val(5),  // fp32_acc
        true                          // pop_inputs
        >;
#endif

    Matmul::Op<CTArgs> matmul;
    matmul();
}
KERNEL_END
