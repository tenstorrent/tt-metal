// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Writer kernel with rt_dim=2 support
// Writes 2 rows of Y at a time to match compute kernel's rt_dim=2 processing
// ============================================================================

#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_y_idx = tt::CBIndex::c_10;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t rt_dim = 2;  // Process 2 rows at a time

void kernel_main() {
    uint32_t ra = 0;
    const uint32_t y_addr = get_arg_val<uint32_t>(ra++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_y_idx);

    constexpr auto y_args = TensorAccessorArgs<2>();
    const auto y_address_generator = TensorAccessor(y_args, y_addr, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;

    // Process rows in pairs (rt_dim=2)
    for (uint32_t r = start_row; r < end_row; r += rt_dim) {
        // Determine actual rows to write (1 or 2 for last pair)
        const uint32_t rows_to_write = (r + rt_dim <= end_row) ? rt_dim : (end_row - r);

        // Y tiles come out in order: for each c_block, all rows' tiles
        // Compute produces: Y[r0, c0], Y[r1, c0], Y[r0, c1], Y[r1, c1], ...
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);

            // Write each row's c_block tiles
            for (uint32_t row_offset = 0; row_offset < rows_to_write; ++row_offset) {
                const uint32_t current_row = r + row_offset;
                const uint32_t start_tile_idx = current_row * Wt + c_block_start;
                write_tiles_by_row(cb_y_idx, y_address_generator, start_tile_idx, c_block_size, tile_bytes, block_size);
            }
        }
    }
}
