// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/tile_move_copy.h"

/**
 * Each core computes: output[M=1, N=1] = in0[M=1, K] @ weight0[K, N=1]
 * where M and N are in tiles, and K is the inner dimension.
 *
 * Both in0 and weight0 are fully available in L1 (sharded tensors).
 *
 * The computation accumulates across the K dimension:
 * for k in range(K):
 *     output += in0[:, k] @ weight0[k, :]
 */
namespace NAMESPACE {

void matmul_with_relu_block(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t num_tiles) {
    cb_wait_front(cb_a, num_tiles);
    cb_wait_front(cb_b, num_tiles);
    cb_reserve_back(cb_out, num_tiles);

    tile_regs_acquire();

    for (uint32_t k = 0; k < num_tiles; k++) {
        matmul_tiles(cb_a, cb_b, k, k, 0);
    }
    relu_tile(0);

    tile_regs_commit();

    // Don't pop front here because we need to use the input again for the next operation
    // cb_pop_front(cb_a, num_tiles);
    // cb_pop_front(cb_b, num_tiles);

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, num_tiles);
}

void matmul_with_bias_block(uint32_t cb_a, uint32_t cb_b, uint32_t cb_bias, uint32_t cb_out, uint32_t num_tiles) {
    cb_wait_front(cb_a, num_tiles);
    cb_wait_front(cb_b, num_tiles);
    cb_wait_front(cb_bias, num_tiles);
    cb_reserve_back(cb_out, num_tiles);

    tile_regs_acquire();

    for (uint32_t k = 0; k < num_tiles; k++) {
        matmul_tiles(cb_a, cb_b, k, k, 0);
    }
    // TODO: Add bias here!

    tile_regs_commit();

    cb_pop_front(cb_a, num_tiles);
    cb_pop_front(cb_b, num_tiles);
    cb_pop_front(cb_bias, num_tiles);

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, num_tiles);
}

void MAIN {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(5);
    constexpr bool fp32_dest_acc_en = get_compile_time_arg_val(6);

    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 1;
    constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

    mm_block_init(in0_cb, weight0_cb, out_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
    relu_tile_init();

    matmul_with_relu_block(in0_cb, weight0_cb, interm_cb, num_tiles_k);
    matmul_with_bias_block(interm_cb, weight1_cb, in0_cb, out_cb, num_tiles_k);
}
}  // namespace NAMESPACE
