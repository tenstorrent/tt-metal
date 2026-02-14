// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_x = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(4);
    constexpr uint32_t total_num_rows = get_compile_time_arg_val(5);

    // Set up tensor accessor
    constexpr auto output_args = TensorAccessorArgs<6>();

    // Common runtime args (shared across all cores)
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    // Calculate work assignment for this core based on coordinates
    const uint32_t core_id_x = get_absolute_logical_x();
    const uint32_t core_id_y = get_absolute_logical_y();
    // Match factory's column-major indexing: core_idx = x * num_cores_y + y
    const uint32_t core_id = core_id_x * num_cores_y + core_id_y;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t tiles_per_core = (total_num_rows + num_cores - 1) / num_cores;

    const uint32_t start_tile = core_id * tiles_per_core;
    const uint32_t end_tile =
        ((start_tile + tiles_per_core) < total_num_rows) ? (start_tile + tiles_per_core) : total_num_rows;
    const uint32_t num_rows = (start_tile < total_num_rows) ? (end_tile - start_tile) : 0;

    // Get tile size
    const uint32_t out_tile_size = get_tile_size(out_cb_id);

    // Create tensor accessor
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_tile_size);

    // Write output rows in blocks
    for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Process tiles in blocks within each row
        for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
            // Calculate block size (handle remainder)
            const uint32_t current_block_size = (block_start + tiles_per_block <= num_tiles_per_row)
                                                    ? tiles_per_block
                                                    : (num_tiles_per_row - block_start);

            write_tiles_by_row(
                out_cb_id,
                output_accessor,
                row_start_tile + block_start,
                current_block_size,
                out_tile_size,
                current_block_size);
        }
    }
}
