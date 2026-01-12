// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"

/**
 * Optimized compute kernel for prepare_expert_weights operation.
 *
 * This version uses more efficient tile operations:
 * 1. Pre-fills a "ones" tile template
 * 2. For each weight, multiplies the ones tile by the scalar
 * 3. Uses tile-level operations instead of element-by-element filling
 *
 * Compile-time args:
 *   0: input_cb - Input circular buffer (weights [B*S, K])
 *   1: scalar_cb - Scalar broadcast buffer
 *   2: output_cb - Output circular buffer
 *   3: ones_cb - Pre-filled ones tile for multiplication
 *   4: batch_seq - B*S dimension
 *   5: num_experts_per_tok - K dimension
 *   6: hidden_size - H dimension
 *   7: num_output_tiles_h - Number of tiles along H dimension
 *   8: input_tile_width - Width of input tiles for indexing
 */

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t ones_cb = get_compile_time_arg_val(3);
    constexpr uint32_t batch_seq = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(5);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_output_tiles_h = get_compile_time_arg_val(7);
    constexpr uint32_t input_tile_width = get_compile_time_arg_val(8);

    // Initialize multiply operation for scalar broadcast
    // We'll multiply ones tile by scalar to broadcast
    mul_tiles_bcast_scalar_init_short(ones_cb, scalar_cb);

    // Wait for input weights and ones tile
    cb_wait_front(input_cb, 1);
    cb_wait_front(ones_cb, 1);

    // Get pointer to input data for direct scalar access
    volatile tt_l1_ptr uint16_t* input_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(input_cb));

    // Process in output order: K, B*S, H
    for (uint32_t k = 0; k < num_experts_per_tok; k++) {
        for (uint32_t bs = 0; bs < batch_seq; bs++) {
            // Read scalar weight from input at position [bs, k]
            uint32_t input_idx = bs * input_tile_width + k;
            uint16_t weight_bf16 = input_ptr[input_idx];

            // Write scalar to scalar_cb for broadcast multiply
            cb_reserve_back(scalar_cb, 1);
            volatile tt_l1_ptr uint16_t* scalar_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scalar_cb));
            // Fill first position (scalar broadcast reads from position 0)
            scalar_ptr[0] = weight_bf16;
            cb_push_back(scalar_cb, 1);

            cb_wait_front(scalar_cb, 1);

            // Reserve output tiles for this (k, bs) pair
            cb_reserve_back(output_cb, num_output_tiles_h);

            // Multiply ones tile by scalar and write to output
            // This effectively broadcasts the scalar across all elements
            tile_regs_acquire();

            for (uint32_t h_tile = 0; h_tile < num_output_tiles_h; h_tile++) {
                // Multiply ones tile by scalar: result = 1.0 * scalar = scalar (broadcast)
                mul_tiles_bcast_scalar(ones_cb, scalar_cb, 0, 0, 0);

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, output_cb);
                tile_regs_release();

                if (h_tile < num_output_tiles_h - 1) {
                    tile_regs_acquire();
                }
            }

            cb_push_back(output_cb, num_output_tiles_h);
            cb_pop_front(scalar_cb, 1);
        }
    }

    cb_pop_front(input_cb, 1);
}
}  // namespace NAMESPACE
