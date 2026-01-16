// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Matmul unified kernel
// Single kernel file, compiles correctly for all RISC cores
//
// NCRISC: Signals sharded CBs are ready
// BRISC: Waits for output tiles
// TRISC: Performs matmul compute via Matmul::Op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

KERNEL_ENTRY {
// ============================================================================
// Define args per RISC (different compile-time arg layout per processor)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type alias (required for Op template)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Named compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("matmul_in0");
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
    constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");

    // Setup sharded persistent buffers (in0 and in1 are backed by L1 shards)
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(in0_cb, num_tiles_k);
        // in1 has num_tiles_k * out_w tiles (K tiles for each output column)
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
    }

    // Reader args (empty - no-op in Matmul::Op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type alias (required for Op template)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    // Writer args (empty - no-op in Matmul::Op)
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type alias (required for Op template) - out_w is compile-time for TRISC
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w")>;

    // Named compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("matmul_in0");
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("matmul_out");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");

    // Compute args
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        .in0 = in0_cb,
        .in1 = in1_cb,
        .out = out_cb,
        .k_num_tiles = num_tiles_k,
    };
#endif

    // ========================================================================
    // Matmul operation
    // CTArgs, IsActiveCore, pop_in0=true, pop_in1=true
    // ========================================================================
    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_active_core, true, true> matmul;
    matmul(matmul_args);
}
KERNEL_END
