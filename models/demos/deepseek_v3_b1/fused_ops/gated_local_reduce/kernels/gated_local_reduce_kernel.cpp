// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Gated Local Reduce fused kernel
//
// Composes two LocalReduce ADD operations plus a final multiplication:
//   Phase 1: reduce(group1) + SiLU -> intermed[0]  (ADD with SiLU)
//   Phase 2: reduce(group2)        -> intermed[1]  (ADD)
//   Phase 3: intermed[0] * intermed[1] -> out      (single mul_tiles)
//
// NCRISC: Signals sharded CBs are ready
// BRISC: No-op
// TRISC: Performs gated local reduce via composed LocalReduce ops + mul_tiles

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/local_reduce.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("gated_local_reduce_in0_cb");
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("gated_local_reduce_in1_cb");
    constexpr uint32_t group1_num_tiles = get_named_compile_time_arg_val("gated_local_reduce_group1_num_tiles");
    constexpr uint32_t group2_num_tiles = get_named_compile_time_arg_val("gated_local_reduce_group2_num_tiles");

    // Setup sharded buffers for both input groups
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(in0_cb, group1_num_tiles);
        unified_kernels::setup_sharded_buffer(in1_cb, group2_num_tiles);
    }

#elif defined(COMPILE_FOR_BRISC)
    // No-op for BRISC

#elif defined(COMPILE_FOR_TRISC)
    if constexpr (Core::is_active_core) {
        // Get compile-time args
        constexpr uint32_t in0_cb = get_named_compile_time_arg_val("gated_local_reduce_in0_cb");
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("gated_local_reduce_in1_cb");
        constexpr uint32_t intermed_cb = get_named_compile_time_arg_val("gated_local_reduce_intermed_cb");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("gated_local_reduce_out_cb");
        constexpr uint32_t group1_num_tiles = get_named_compile_time_arg_val("gated_local_reduce_group1_num_tiles");
        constexpr uint32_t group2_num_tiles = get_named_compile_time_arg_val("gated_local_reduce_group2_num_tiles");
        deepseek_compute_kernel_init();

        // ================================================================
        // Phase 1: reduce(group1) + SiLU -> intermed[0]  (ADD with SiLU)
        // ================================================================
        using Group1CTArgs = deepseek_b1_ops::LocalReduce::ComputeCTArgs<group1_num_tiles, true>;
        deepseek_b1_ops::LocalReduce::ComputeArgs group1_args{in0_cb, intermed_cb};
        deepseek_b1_ops::LocalReduce::Op<Group1CTArgs, true> group1_reduce;
        group1_reduce(group1_args);

        // ================================================================
        // Phase 2: reduce(group2) -> intermed[1]  (ADD)
        // ================================================================
        using Group2CTArgs = deepseek_b1_ops::LocalReduce::ComputeCTArgs<group2_num_tiles, false>;
        deepseek_b1_ops::LocalReduce::ComputeArgs group2_args{in1_cb, intermed_cb};
        deepseek_b1_ops::LocalReduce::Op<Group2CTArgs, true> group2_reduce;
        group2_reduce(group2_args);

        // ================================================================
        // Phase 3: intermed[0] * intermed[1] -> out  (single mul_tiles)
        // ================================================================
        // Wait for both intermediate tiles
        cb_wait_front(intermed_cb, 2);

        // Reserve output
        cb_reserve_back(out_cb, 1);

        // Initialize and perform multiplication
        binary_op_init_common(intermed_cb, intermed_cb, out_cb);
        tile_regs_acquire();
        mul_tiles_init(intermed_cb, intermed_cb);
        mul_tiles(intermed_cb, intermed_cb, 0, 1, 0);

        // Commit and wait
        tile_regs_commit();
        tile_regs_wait();

        // Pack result
        pack_tile(0, out_cb);
        tile_regs_release();

        // Pop intermediates and push output
        cb_pop_front(intermed_cb, 2);
        cb_push_back(out_cb, 1);
    }
#endif
}
