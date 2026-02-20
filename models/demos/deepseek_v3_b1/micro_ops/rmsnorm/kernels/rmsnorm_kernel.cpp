// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// RMSNorm unified kernel
// Single kernel file, compiles correctly for all RISC cores
//
// NCRISC: Sets up sharded buffers and generates reduction scalar
// BRISC: No-op (waits are handled by next op in pipeline)
// TRISC: Performs RMSNorm compute via RMSNorm::Op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define args per RISC (different compile-time arg layout per processor)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type alias (required for Op template)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;

    // Named compile-time args
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

    // Setup sharded persistent buffers (input and gamma are backed by L1 shards)
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(gamma_cb, num_tiles);
        unified_kernels::setup_sharded_buffer(input_cb, num_tiles);
    }

    // Reader args: none needed
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type alias (required for Op template)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;

    // Writer args (empty - BRISC is no-op for RMSNorm)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

#elif defined(COMPILE_FOR_TRISC)
    // Named compile-time args
    constexpr uint32_t input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("rmsnorm_output_cb");

    // CTArgs type alias (required for Op template)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        input_cb,
        gamma_cb,
        output_cb>;

    // Compute args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        .epsilon = get_common_arg_val<uint32_t>(0),  // epsilon
        .scalar = get_common_arg_val<float>(1),      // scalar (1/sqrt(num_elements))
    };
    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif

    // ========================================================================
    // RMSNorm operation
    // CTArgs, IsActiveCore, pop_input=true
    // ========================================================================
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_active_core, true> rmsnorm;
    rmsnorm(rmsnorm_args);
}
