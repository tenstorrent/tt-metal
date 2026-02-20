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
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/rope.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define args per RISC (different compile-time arg layout per processor)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr uint32_t cos_sin_page_size = get_named_compile_time_arg_val("cos_sin_page_size");
    constexpr uint32_t total_Wt = get_named_compile_time_arg_val("total_Wt");
    constexpr uint32_t start_tile_offset = get_named_compile_time_arg_val("start_tile_offset");
    using RopeCTArgs = deepseek_b1_ops::Rope::ReaderCTArgs<Wt, Ht, cos_sin_page_size, total_Wt, start_tile_offset>;

    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t cos_cb = get_named_compile_time_arg_val("cos_cb");
    constexpr uint32_t sin_cb = get_named_compile_time_arg_val("sin_cb");
    constexpr uint32_t cos_tensor_address = get_named_compile_time_arg_val("cos_tensor_address");
    constexpr uint32_t sin_tensor_address = get_named_compile_time_arg_val("sin_tensor_address");
    constexpr uint32_t position_ids_tensor_address = get_named_compile_time_arg_val("position_ids_tensor_address");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");

    unified_kernels::setup_sharded_buffer(in_cb, Wt);
    unified_kernels::setup_sharded_buffer(trans_mat_cb, 1);

    deepseek_b1_ops::Rope::ReaderArgs rope_args{
        .in_cb = in_cb,
        .cos_cb = cos_cb,
        .sin_cb = sin_cb,
        .cos_tensor_address = cos_tensor_address,
        .sin_tensor_address = sin_tensor_address,
        .position_ids_tensor_address = position_ids_tensor_address,
        .trans_mat_cb = trans_mat_cb,
    };

#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type alias (required for Op template)
    using RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;

    // Writer args (empty - no-op)
    deepseek_b1_ops::Rope::WriterArgs rope_args{};

#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type alias
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    using RopeCTArgs = deepseek_b1_ops::Rope::ComputeCTArgs<Wt, Ht>;

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
    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);  // Full init, CBs don't matter
#endif

    // ========================================================================
    // RoPE operation
    // CTArgs, IsActiveCore
    // ========================================================================
    deepseek_b1_ops::Rope::Op<RopeCTArgs, Core::is_active_core> rope;
    rope(rope_args);
}
