// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of deepseek_v3_b1 GatedReduce's compute with the per-K scalar (the SRAM path of moe_kernel.cpp
// and decoder_block_kernel.cpp: enable_scalar = enable_routing): models/demos/deepseek_v3_b1/unified_kernels/gated_reduce.hpp
// :121-207 verbatim, after the fused kernels' deepseek_compute_kernel_init, run TWIN_ITERS times.
// Compile args: group1_cb, group2_cb, intermed_cb, out_cb, scalar_cb, tiles_per_k, k_num_tiles, iterations.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/eltwise_mul_scalar.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"

struct GrArgs {
    uint32_t group1_cb;
    uint32_t group2_cb;
    uint32_t intermed_cb;
    uint32_t out_cb;
    uint32_t scalar_cb;
    uint32_t k_num_tiles;
    uint32_t out_cb_total_pushes;
};

template <uint32_t tiles_per_k>
void gated_reduce_scalar(const GrArgs& args) {
    const uint32_t k_num_tiles = args.k_num_tiles;
    static_assert(tiles_per_k >= 2 && tiles_per_k % 2 == 0, "tiles_per_k must be even and >= 2");

    // Init once before the loop
    // Assumes all input cbs are configured the same, and the intermediate cb is configured the same as the
    // output cb
    reconfig_full_operand(args.group1_cb, args.group1_cb);
    pack_reconfig_data_format<true>(args.out_cb);
    silu_tile_init();
    for (uint32_t k = 0; k < k_num_tiles; k++) {
        // Group 1: reduce + SiLU
        add_init(args.group1_cb, args.group1_cb, true /* acc_to_dest */);

        cb_wait_front(args.group1_cb, tiles_per_k);
        cb_reserve_back(args.intermed_cb, 1);

        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_k; i += 2) {
            add_tiles(args.group1_cb, args.group1_cb, i, i + 1, 0);
        }
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, args.intermed_cb);
        tile_regs_release();

        cb_pop_front(args.group1_cb, tiles_per_k);
        cb_push_back(args.intermed_cb, 1);

        // Group 2: reduce (skip re-init add for different CB assuming they're configured the same)

        cb_wait_front(args.group2_cb, tiles_per_k);
        cb_reserve_back(args.intermed_cb, 1);

        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_k; i += 2) {
            add_tiles(args.group2_cb, args.group2_cb, i, i + 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, args.intermed_cb);
        tile_regs_release();

        cb_pop_front(args.group2_cb, tiles_per_k);
        cb_push_back(args.intermed_cb, 1);

        // Multiply: out = SiLU(g1) * scale[k] * g2  (SRAM path)
        cb_wait_front(args.intermed_cb, 2);
        cb_reserve_back(args.out_cb, 1);

        // Wait for this iteration's scalar (BRISC pushes one bf16 per
        // active expert into scalar_cb in TopK SRAM-flagged order).
        cb_wait_front(args.scalar_cb, 1);

        tile_regs_acquire();
        // DST[0] = silu(g1) * scale[0]
        deepseek_mul_bcast_scalar_init(args.intermed_cb, args.scalar_cb);
        deepseek_mul_tiles_bcast_scalar(args.intermed_cb, args.scalar_cb, 0, 0, 0);
        // DST[0] *= sum(g2)
        deepseek_binary_dest_reuse_tiles_init(args.intermed_cb);
        deepseek_binary_dest_reuse_tiles(args.intermed_cb, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, args.out_cb);
        tile_regs_release();

        cb_pop_front(args.scalar_cb, 1);

        cb_pop_front(args.intermed_cb, 2);
        cb_push_back(args.out_cb, 1);
    }

    const uint32_t padding = (args.out_cb_total_pushes > k_num_tiles) ? (args.out_cb_total_pushes - k_num_tiles) : 0;
    if (padding > 0) {
        cb_reserve_back(args.out_cb, padding);
        cb_push_back(args.out_cb, padding);
    }
}

void kernel_main() {
    constexpr uint32_t tiles_per_k = get_compile_time_arg_val(5);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(7);
    const GrArgs args{
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(6),
    };

    deepseek_compute_kernel_init();
    for (uint32_t it = 0; it < twin_iters; ++it) {
        gated_reduce_scalar<tiles_per_k>(args);
    }
}
