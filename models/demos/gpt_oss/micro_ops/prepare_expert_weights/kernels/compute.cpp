// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/bcast.h"

/**
 * Compute kernel for prepare_expert_weights operation.
 *
 * Transforms routing weights from [B*S, K] to [K, 1, B*S, H] format by:
 * 1. Reading each scalar weight from input
 * 2. Broadcasting it across the hidden dimension
 * 3. Writing output tiles in the correct permuted order
 *
 * The output layout is [K, 1, B*S, H] which means:
 * - Outer loop: K experts
 * - Middle loop: B*S batch*sequence positions
 * - Inner loop: H hidden dimension (tiled)
 *
 * Each weight value input[bs, k] is broadcast to output[k, 0, bs, 0:H].
 *
 * Compile-time args:
 *   0: input_cb - Input circular buffer (weights [B*S, K])
 *   1: scalar_cb - Scalar broadcast buffer (single tile)
 *   2: output_cb - Output circular buffer ([K, 1, B*S, H])
 *   3: batch_seq - B*S dimension
 *   4: num_experts_per_tok - K dimension
 *   5: hidden_size - H dimension
 *   6: num_output_tiles_h - Number of tiles along H dimension
 *   7: input_tile_width - Width of input tiles for indexing
 */

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t batch_seq = get_compile_time_arg_val(3);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(4);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_output_tiles_h = get_compile_time_arg_val(6);
    constexpr uint32_t input_tile_width = get_compile_time_arg_val(7);

    // Initialize copy operation for broadcasting
    copy_tile_to_dst_init_short(input_cb);

    // Wait for input weights to be ready
    cb_wait_front(input_cb, 1);  // Input is in a single tile

    // Get pointer to input data for direct scalar access
    // Input layout: [B*S, K] packed in row-major within tile
    volatile tt_l1_ptr uint16_t* input_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(input_cb));

    // Process in output order: K, B*S, H
    // This ensures output tiles are written in the correct order
    for (uint32_t k = 0; k < num_experts_per_tok; k++) {
        for (uint32_t bs = 0; bs < batch_seq; bs++) {
            // Read scalar weight from input at position [bs, k]
            // Input is stored row-major: index = bs * input_tile_width + k
            uint32_t input_idx = bs * input_tile_width + k;
            uint16_t weight_bf16 = input_ptr[input_idx];

            // Reserve space in scalar CB for broadcast tile
            cb_reserve_back(scalar_cb, 1);

            // Fill the scalar CB tile with the weight value
            // This creates a tile where every element = weight_bf16
            volatile tt_l1_ptr uint16_t* scalar_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scalar_cb));

            // Fill entire tile with the scalar value
            // Tile is 32x32 = 1024 elements for standard tiles
            // For bfloat16, we fill with the same value
            constexpr uint32_t tile_elements = 32 * 32;  // Assuming 32x32 tiles
            for (uint32_t i = 0; i < tile_elements; i++) {
                scalar_ptr[i] = weight_bf16;
            }

            cb_push_back(scalar_cb, 1);

            // Now copy this scalar tile to each output tile along H dimension
            cb_wait_front(scalar_cb, 1);

            // Reserve output tiles for this (k, bs) pair
            cb_reserve_back(output_cb, num_output_tiles_h);

            tile_regs_acquire();

            for (uint32_t h_tile = 0; h_tile < num_output_tiles_h; h_tile++) {
                // Copy scalar tile to dst register
                copy_tile(scalar_cb, 0, 0);

                // Pack to output
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, output_cb);
                tile_regs_release();

                if (h_tile < num_output_tiles_h - 1) {
                    tile_regs_acquire();
                }
            }

            cb_push_back(output_cb, num_output_tiles_h);

            // Pop the scalar tile so we can reuse the CB
            cb_pop_front(scalar_cb, 1);
        }
    }

    // Pop input (we're done with it)
    cb_pop_front(input_cb, 1);
}
}  // namespace NAMESPACE
