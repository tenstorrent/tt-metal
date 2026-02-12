// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// ScatterHeads unified kernel
// Single kernel file, compiles correctly for all RISC cores (NCRISC, BRISC, TRISC)
// Note: This is a dataflow-only op - TRISC is a no-op
//
// Scatters data from 8 input cores (each with 8 rows) to 64 output cores (each with 1 row)
// - NCRISC: Reader on output cores - performs noc_async_read from source core
// - BRISC: No-op
// - TRISC: No-op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/scatter_heads.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_output_core = get_named_compile_time_arg_val("is_output_core") == 1;
};

void kernel_main() {
    using ScatterHeads = deepseek_b1_ops::ScatterHeads;

// ============================================================================
// NCRISC (Reader) - Output cores read from input cores
// Named compile-time args: scatter reader params
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Reader args for output cores
    ScatterHeads::ReaderArgs scatter_args{
        get_named_compile_time_arg_val("src_noc_x"),
        get_named_compile_time_arg_val("src_noc_y"),
        get_named_compile_time_arg_val("src_addr"),
        get_named_compile_time_arg_val("src_row_offset"),
        get_named_compile_time_arg_val("data_size_bytes"),
        get_named_compile_time_arg_val("dst_cb"),
        get_named_compile_time_arg_val("dst_num_pages"),
    };

// ============================================================================
// BRISC (Writer) - No-op
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Writer args for input cores
    ScatterHeads::WriterArgs scatter_args{};

// ============================================================================
// TRISC (Compute) - No-op
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    ScatterHeads::ComputeArgs scatter_args = {};

#endif

    // Execute scatter operation (NCRISC and BRISC only)
    ScatterHeads::Op<Core::is_output_core> scatter;
    scatter(scatter_args);
}
