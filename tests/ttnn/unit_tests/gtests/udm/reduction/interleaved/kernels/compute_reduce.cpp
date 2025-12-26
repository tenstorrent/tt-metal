// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Compute Reduce - Interleaved Version
 *
 * Performs width reduction (SUM) on tiles from input circular buffer.
 * Reduces a row of tiles (width tiles) into a single output tile.
 *
 * Runtime args (same pattern as add kernel):
 *   - rank: number of dimensions
 *   - For each dimension: (num_pages, offset, stride)
 *   Note: num_rows = product of all dims except last, input_width_tiles = last dim
 *
 * CB layout:
 *   - CB 0: Input tiles (row of width tiles)
 *   - CB 1: Scaler tile (1.0 for SUM)
 *   - CB 2: Output tiles (1 reduced tile per row)
 */

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#define FLOAT32_REDUCTION true

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // ==================== Get Runtime Arguments ====================
    // Compute only needs: rank, then pages per dim (no offset/stride needed)
    uint32_t rank = get_arg_val<uint32_t>(0);

    constexpr uint32_t MAX_RANK = 8;
    uint32_t dim_pages[MAX_RANK] = {0};

    for (uint32_t d = 0; d < rank; ++d) {
        dim_pages[d] = get_arg_val<uint32_t>(1 + d);
    }

    // Calculate num_rows (product of all dims except last) and input_width_tiles (last dim)
    uint32_t num_rows = 1;
    for (uint32_t d = 0; d < rank - 1; ++d) {
        num_rows *= dim_pages[d];
    }
    uint32_t input_width_tiles = dim_pages[rank - 1];

    // Early exit for non-workers
    if (num_rows == 0 || input_width_tiles == 0) {
        return;
    }

    // ==================== CB definitions ====================
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;     // Input tiles
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;  // Scaler
    constexpr uint32_t cb_out = tt::CBIndex::c_2;     // Output tiles

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    // ==================== Initialize reduction ====================
    binary_op_init_common(cb_in0, cb_in0, cb_out);
    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in0, cb_scaler, cb_out);

    // Wait for scaler to be ready (pushed by dataflow)
    cb_wait_front(cb_scaler, 1);

    // ==================== Process each row ====================
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for input row tiles
        cb_wait_front(cb_in0, input_width_tiles);

        // Acquire destination register
        tile_regs_acquire();

        // Reduce across width for this row
        for (uint32_t col = 0; col < input_width_tiles; ++col) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in0, cb_scaler, col, scaler0, dst0);
        }

        // Commit and pack result
        tile_regs_commit();

        // Reserve output space
        cb_reserve_back(cb_out, 1);

        tile_regs_wait();
        pack_tile(dst0, cb_out);
        tile_regs_release();

        // Push output and pop input
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, input_width_tiles);
    }

    // Cleanup
    reduce_uninit();
}
}  // namespace NAMESPACE
