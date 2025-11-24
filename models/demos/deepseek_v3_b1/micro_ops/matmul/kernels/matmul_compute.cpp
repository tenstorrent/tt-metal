// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

/**
 * Simplified matmul compute kernel for single output tile per core.
 *
 * Each core computes: output[M=1, N=1] = in0[M=1, K] @ in1[K, N=1]
 * where M and N are in tiles, and K is the inner dimension.
 *
 * Both in0 and in1 are fully available in L1 (sharded tensors).
 *
 * The computation accumulates across the K dimension:
 * for k in range(K):
 *     output += in0[:, k] @ in1[k, :]
 */
namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(4);
    constexpr bool fp32_dest_acc_en = get_compile_time_arg_val(5);

    // For single tile output, we use simple matmul accumulation
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 1;
    constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

    // Initialize matmul
    mm_block_init(in0_cb, in1_cb, out_cb, false, out_subblock_w, out_subblock_h, in0_block_w);

    // Wait for all input tiles (both from sharded tensors in L1)
    cb_wait_front(in0_cb, num_tiles_k);
    cb_wait_front(in1_cb, num_tiles_k);

    // Reserve output
    cb_reserve_back(out_cb, 1);

    // Accumulate across K dimension
    tile_regs_acquire();

    for (uint32_t k = 0; k < num_tiles_k; k++) {
        // Compute matmul for this k tile
        // in0 tile index: k (from sharded input)
        // in1 tile index: k (from sharded weights)
        // dst index: 0 (single output tile, accumulating)
        matmul_tiles(in0_cb, in1_cb, k, k, 0, false);
    }

    tile_regs_commit();

    // Pop inputs
    cb_pop_front(in0_cb, num_tiles_k);
    cb_pop_front(in1_cb, num_tiles_k);

    // Pack output
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();

    cb_push_back(out_cb, 1);
}
}  // namespace NAMESPACE
