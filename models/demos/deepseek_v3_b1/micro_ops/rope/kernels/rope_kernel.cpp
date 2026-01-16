// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// RoPE unified kernel
// Single kernel file, compiles correctly for all RISC cores
//
// Computes: output = (input * cos) + (rotate_half(input) * sin)
// where rotate_half(input) = input @ trans_mat
//
// NCRISC: Signals sharded input CBs (input, trans_mat, sin, cos)
// BRISC: No-op
// TRISC: Performs RoPE compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/rope.hpp"

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
    using RopeCTArgs = deepseek_b1_ops::Rope::ReaderCTArgs<get_named_compile_time_arg_val("Wt")>;

    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");

    // Reader args: CB indices for sharded input signaling
    deepseek_b1_ops::Rope::ReaderArgs rope_args{
        .in_cb = in_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .trans_mat_cb = trans_mat_cb,
    };

#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type alias (required for Op template)
    using RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;

    // Writer args (empty - no-op)
    deepseek_b1_ops::Rope::WriterArgs rope_args{};

#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type alias with Wt as template parameter (Ht=1 hardcoded)
    using RopeCTArgs = deepseek_b1_ops::Rope::ComputeCTArgs<get_named_compile_time_arg_val("Wt")>;

    // CB indices (passed as runtime args to ComputeArgs)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
    constexpr uint32_t rotated_in_interm_cb = get_named_compile_time_arg_val("rotated_in_interm_cb");
    constexpr uint32_t cos_interm_cb = get_named_compile_time_arg_val("cos_interm_cb");
    constexpr uint32_t sin_interm_cb = get_named_compile_time_arg_val("sin_interm_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");

    // Compute args: all CB indices
    deepseek_b1_ops::Rope::ComputeArgs rope_args{
        .in_cb = in_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .trans_mat_cb = trans_mat_cb,
        .rotated_in_interm_cb = rotated_in_interm_cb,
        .cos_interm_cb = cos_interm_cb,
        .sin_interm_cb = sin_interm_cb,
        .out_cb = out_cb,
    };
#endif

    // ========================================================================
    // RoPE operation
    // CTArgs, IsActiveCore
    // ========================================================================
    deepseek_b1_ops::Rope::Op<RopeCTArgs, Core::is_active_core> rope;
    rope(rope_args);
}
KERNEL_END
