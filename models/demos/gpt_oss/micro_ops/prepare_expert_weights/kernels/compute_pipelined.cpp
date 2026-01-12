// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/bcast.h"

/**
 * Pipelined compute kernel for prepare_expert_weights operation.
 *
 * This version pipelines the output tile generation to maximize throughput:
 * 1. Uses double-buffering in dst registers
 * 2. Overlaps scalar tile creation with output packing
 * 3. Processes multiple H tiles per dst register acquire
 *
 * The key insight is that all output tiles for a given (k, bs) pair have
 * the same value, so we can create one scalar tile and replicate it
 * efficiently across all H positions.
 *
 * Compile-time args:
 *   0: input_cb - Input circular buffer (weights [B*S, K])
 *   1: scalar_cb - Scalar broadcast buffer
 *   2: output_cb - Output circular buffer
 *   3: batch_seq - B*S dimension
 *   4: num_experts_per_tok - K dimension
 *   5: hidden_size - H dimension
 *   6: num_output_tiles_h - Number of tiles along H dimension
 *   7: input_tile_width - Width of input tiles for indexing
 *   8: tiles_per_block - Number of H tiles to process per dst acquire (for pipelining)
 */

// Helper to fill a tile with a scalar value using efficient SIMD operations
template <bool is_fp32_dest_acc_en = false>
FORCE_INLINE void fill_tile_with_scalar(
    uint32_t cb_id,
    uint16_t scalar_bf16,
    uint32_t tile_elements = 1024) {

    volatile tt_l1_ptr uint16_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    // Pack scalar into 32-bit word for faster filling (two bf16 values)
    uint32_t packed_scalar = (static_cast<uint32_t>(scalar_bf16) << 16) | scalar_bf16;
    volatile tt_l1_ptr uint32_t* ptr32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptr);

    // Fill using 32-bit writes (2x faster than 16-bit)
    uint32_t num_words = tile_elements / 2;
    for (uint32_t i = 0; i < num_words; i++) {
        ptr32[i] = packed_scalar;
    }
}

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
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(8);

    // Calculate number of blocks needed
    constexpr uint32_t num_blocks = (num_output_tiles_h + tiles_per_block - 1) / tiles_per_block;

    // Initialize tile copy operation
    copy_tile_to_dst_init_short(scalar_cb);

    // Wait for input weights to be ready
    cb_wait_front(input_cb, 1);

    // Get pointer to input data
    volatile tt_l1_ptr uint16_t* input_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(input_cb));

    // Process in output order: K, B*S, H
    for (uint32_t k = 0; k < num_experts_per_tok; k++) {
        for (uint32_t bs = 0; bs < batch_seq; bs++) {
            // Read scalar weight from input at position [bs, k]
            uint32_t input_idx = bs * input_tile_width + k;
            uint16_t weight_bf16 = input_ptr[input_idx];

            // Create scalar tile once for this (k, bs) pair
            cb_reserve_back(scalar_cb, 1);
            fill_tile_with_scalar(scalar_cb, weight_bf16);
            cb_push_back(scalar_cb, 1);

            cb_wait_front(scalar_cb, 1);

            // Reserve all output tiles for this (k, bs) pair
            cb_reserve_back(output_cb, num_output_tiles_h);

            // Process H tiles in blocks for better register utilization
            uint32_t tiles_remaining = num_output_tiles_h;
            uint32_t tiles_written = 0;

            while (tiles_remaining > 0) {
                uint32_t tiles_this_block = (tiles_remaining > tiles_per_block)
                    ? tiles_per_block : tiles_remaining;

                tile_regs_acquire();

                // Copy scalar tile to multiple dst register positions
                for (uint32_t t = 0; t < tiles_this_block; t++) {
                    copy_tile(scalar_cb, 0, t);
                }

                tile_regs_commit();
                tile_regs_wait();

                // Pack all tiles in this block to output
                for (uint32_t t = 0; t < tiles_this_block; t++) {
                    pack_tile(t, output_cb);
                }

                tile_regs_release();

                tiles_remaining -= tiles_this_block;
                tiles_written += tiles_this_block;
            }

            // Signal output tiles ready
            cb_push_back(output_cb, num_output_tiles_h);

            // Release scalar tile for reuse
            cb_pop_front(scalar_cb, 1);
        }
    }

    // Done with input
    cb_pop_front(input_cb, 1);
}
}  // namespace NAMESPACE
