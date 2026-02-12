// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t DST_REG_ID = 0;
constexpr uint32_t ONE_TILE = 1;

// Caller is responsible for waiting cb_wait_front(icb_ones, onetile);
ALWI void reduce_tile_to_cb(uint32_t icb0, uint32_t icb_ones, uint32_t ocb, uint32_t size) {
    tile_regs_acquire();

    // Initialize matmul - will accumulate across all tiles
    mm_init(icb0, icb_ones, ocb, /*transpose*/ 0);

    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        // Multiply tile x with ones vector and accumulate to dst0
        // false = no transpose, enables accumulation behavior
        matmul_tiles(icb0, icb_ones, x, 0, DST_REG_ID);
    }

    tile_regs_commit();
    pack_and_push(DST_REG_ID, ocb);
}

// Multiply two tiles from CBs
// Note: caller must ensure tiles are available via cb_wait_front before calling
ALWI void elementwise_multiply(
    uint32_t src0_cb_id, uint32_t src1_cb_id, uint32_t out_cb_id, uint32_t tile_index_0, uint32_t tile_index_1) {
    tile_regs_acquire();
    // Multiply src0_cb_id * src1_cb_id
    mul_tiles(src0_cb_id, src1_cb_id, tile_index_0, tile_index_1, DST_REG_ID);
    tile_regs_commit();
    pack_and_push(DST_REG_ID, out_cb_id);
}

// Create and push a zero tile to the specified circular buffer
ALWI void push_zero_tile(uint32_t cb_id) {
    fill_tile_init();
    tile_regs_acquire();
    fill_tile(DST_REG_ID, 0.0f);
    tile_regs_commit();
    pack_and_push(DST_REG_ID, cb_id);
}

// Add a new value to an accumulator and replace the accumulator with the result
// Reads from accum_cb and addend_cb, pops both, then pushes result back to accum_cb
ALWI void accumulate(uint32_t accum_cb_id, uint32_t addend_cb_id) {
    // Wait for both inputs
    cb_wait_front(accum_cb_id, ONE_TILE);
    cb_wait_front(addend_cb_id, ONE_TILE);

    // Add them together
    add_tiles_init(accum_cb_id, addend_cb_id);
    tile_regs_acquire();
    add_tiles(accum_cb_id, addend_cb_id, 0, 0, DST_REG_ID);
    tile_regs_commit();
    tile_regs_wait();

    // Pop old values
    cb_pop_front(accum_cb_id, ONE_TILE);
    cb_pop_front(addend_cb_id, ONE_TILE);

    // Write updated sum back to accumulator CB
    pack_and_push(DST_REG_ID, accum_cb_id);
}

// Fused subtract and multiply: output = y * (grad - sum)
// Reuses DST register to eliminate intermediate CB write/read
ALWI void fused_sub_mul(
    uint32_t y_cb_id,           // y (softmax output)
    uint32_t grad_cb_id,        // grad (upstream gradient)
    uint32_t sum_reduce_cb_id,  // sum(y * grad) - broadcasted scalar
    uint32_t out_cb_id,         // output
    uint32_t y_tile_idx,
    uint32_t grad_tile_idx) {
    tile_regs_acquire();

    // Step 1: Compute grad - sum(y * grad) and store in DST[0]
    sub_bcast_cols_init_short(grad_cb_id, sum_reduce_cb_id);
    sub_tiles_bcast<BROADCAST_TYPE>(grad_cb_id, sum_reduce_cb_id, grad_tile_idx, 0, DST_REG_ID);

    // Step 2: Multiply y * DST[0], reusing the DST register
    binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(y_cb_id);
    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(y_cb_id, y_tile_idx, DST_REG_ID);

    tile_regs_commit();
    pack_and_push(DST_REG_ID, out_cb_id);
}

void kernel_main() {
    // Compile time args
    constexpr uint32_t y_cb_id = get_compile_time_arg_val(0);            // softmax_output (y)
    constexpr uint32_t grad_cb_id = get_compile_time_arg_val(1);         // upstream_grad (grad)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);          // output
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(3);          // y * grad
    constexpr uint32_t sum_reduce_cb_id = get_compile_time_arg_val(4);   // sum(y * grad) - accumulated
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(5);         // ones vector for matmul reduction
    constexpr uint32_t block_sum_cb_id = get_compile_time_arg_val(6);    // block sum temporary
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(7);  // width in tiles
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(8);  // block size - must match reader/writer kernels
    constexpr bool full_row_in_l1 = (num_tiles_per_row == tiles_per_block);  // skip second read when full row fits

    // Runtime args
    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // Number of rows to process

    // Initialize compute operations
    binary_op_init_common(y_cb_id, grad_cb_id, out_cb_id);

    // Two-pass streaming algorithm for minimal L1 memory
    for (uint32_t row = 0; row < num_rows; ++row) {
        // === PASS 1: Streaming reduction to compute sum(y * grad) ===
        cb_wait_front(ones_cb_id, ONE_TILE);

        // Initialize accumulator with a zero tile
        push_zero_tile(sum_reduce_cb_id);

        // Process in blocks: compute products, then accumulate each block
        for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
            const uint32_t current_block_size = std::min(tiles_per_block, num_tiles_per_row - block_start);

            // Wait for this block from reader
            cb_wait_front(y_cb_id, current_block_size);
            cb_wait_front(grad_cb_id, current_block_size);

            // Step 1a: Compute y * grad for all tiles in this block (elementwise multiplication)
            mul_tiles_init(y_cb_id, grad_cb_id);
            for (uint32_t i = 0; i < current_block_size; ++i) {
                elementwise_multiply(y_cb_id, grad_cb_id, mul_cb_id, i, i);
            }

            // Step 1b: Reduce this block to a single sum tile using matmul with ones
            // Write block sum to temporary CB
            reduce_tile_to_cb(mul_cb_id, ones_cb_id, block_sum_cb_id, current_block_size);

            // Step 1c: Add block sum to running total
            // accumulated_sum = accumulated_sum + block_sum
            accumulate(sum_reduce_cb_id, block_sum_cb_id);

            // Pop this block
            cb_pop_front(mul_cb_id, current_block_size);
            if constexpr (!full_row_in_l1) {
                cb_pop_front(y_cb_id, current_block_size);
                cb_pop_front(grad_cb_id, current_block_size);
            }
        }

        // === PASS 2: Compute final output (reuse data when full row in L1, else fresh read) ===
        cb_wait_front(sum_reduce_cb_id, ONE_TILE);

        for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
            const uint32_t current_block_size = (block_start + tiles_per_block <= num_tiles_per_row)
                                                    ? tiles_per_block
                                                    : (num_tiles_per_row - block_start);

            if constexpr (!full_row_in_l1) {
                // Wait for fresh block from reader (pass 2 read)
                cb_wait_front(y_cb_id, current_block_size);
                cb_wait_front(grad_cb_id, current_block_size);
            }

            // Process each tile: compute y * (grad - sum)
            for (uint32_t i = 0; i < current_block_size; ++i) {
                fused_sub_mul(
                    y_cb_id,           // y
                    grad_cb_id,        // grad
                    sum_reduce_cb_id,  // sum(y * grad)
                    out_cb_id,         // output
                    i,                 // y tile index (relative to block)
                    i);                // grad tile index (relative to block)
            }

            cb_pop_front(y_cb_id, current_block_size);
            cb_pop_front(grad_cb_id, current_block_size);
        }

        // Pop sum for this row
        cb_pop_front(sum_reduce_cb_id, ONE_TILE);
    }
}
