// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

// BF16(1.0) packed twice into u32 for matmul-based reduction
constexpr uint32_t BF16_ONE_PACKED = 0x3f803f80;

// BF16(0.0) for masking
constexpr uint16_t BF16_ZERO = 0x0000;

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(2);  // ones vector for matmul reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(4);  // block size - must match compute/writer kernels
    constexpr uint32_t mask_w = get_compile_time_arg_val(5);           // padding mask width (0 if no padding)
    constexpr uint32_t num_cores_x = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(7);
    constexpr uint32_t total_num_rows = get_compile_time_arg_val(8);
    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<9>();
    constexpr auto upstream_grad_args = TensorAccessorArgs<10>();

    // Common runtime args (shared across all cores)
    const uint32_t softmax_output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t upstream_grad_addr = get_common_arg_val<uint32_t>(1);

    // Calculate work assignment for this core based on coordinates
    const uint32_t core_id_x = get_absolute_logical_x();
    const uint32_t core_id_y = get_absolute_logical_y();
    // Match factory's column-major indexing: core_idx = x * num_cores_y + y
    const uint32_t core_id = core_id_x * num_cores_y + core_id_y;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t rows_per_core = (total_num_rows + num_cores - 1) / num_cores;

    const uint32_t start_row = core_id * rows_per_core;
    const uint32_t end_row =
        ((start_row + rows_per_core) < total_num_rows) ? (start_row + rows_per_core) : total_num_rows;
    const uint32_t num_rows = (start_row < total_num_rows) ? (end_row - start_row) : 0;

    // Get tile sizes
    const uint32_t src0_tile_size = get_tile_size(src0_cb_id);
    const uint32_t src1_tile_size = get_tile_size(src1_cb_id);

    // Create tensor accessors
    const auto softmax_output_accessor = TensorAccessor(softmax_output_args, softmax_output_addr, src0_tile_size);
    const auto upstream_grad_accessor = TensorAccessor(upstream_grad_args, upstream_grad_addr, src1_tile_size);

    // Generate a column vector of ones for matmul-based reduction (BF16 only)
    generate_mm_scaler(ones_cb_id, BF16_ONE_PACKED);

    // When full row fits in L1 (num_tiles_per_row == tiles_per_block), read once; else stream and read twice
    constexpr bool full_row_in_l1 = (num_tiles_per_row == tiles_per_block);
    constexpr uint32_t num_passes = full_row_in_l1 ? 1 : 2;

    for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const uint32_t row_start_tile = (start_row + row_idx) * num_tiles_per_row;

        for (uint32_t pass = 0; pass < num_passes; ++pass) {
            for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
                const uint32_t current_block_size = std::min(tiles_per_block, num_tiles_per_row - block_start);

                cb_reserve_back(src0_cb_id, current_block_size);
                const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);

                cb_reserve_back(src1_cb_id, current_block_size);
                const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);

                for (uint32_t i = 0; i < current_block_size; ++i) {
                    const uint32_t curr_tile = row_start_tile + block_start + i;

                    noc_async_read_page(curr_tile, softmax_output_accessor, l1_write_addr_src0 + i * src0_tile_size);
                    noc_async_read_page(curr_tile, upstream_grad_accessor, l1_write_addr_src1 + i * src1_tile_size);
                }

                // Wait for all tiles in block to be read before masking
                noc_async_read_barrier();

                // Mask padded region in last tile of the row by replacing padded values with 0.0 (BF16 only)
                if constexpr (mask_w > 0) {
                    const uint32_t last_tile_idx_in_block = current_block_size - 1;
                    const uint32_t global_last_tile_idx = block_start + current_block_size - 1;
                    const bool block_contains_last_tile = (global_last_tile_idx == num_tiles_per_row - 1);

                    if (block_contains_last_tile) {
                        // Zero-fill padding in last tile (width-only; full tile height 32)
                        fill_pad_tile<uint16_t, mask_w, 32>(
                            l1_write_addr_src0 + last_tile_idx_in_block * src0_tile_size,
                            static_cast<uint16_t>(BF16_ZERO));
                        fill_pad_tile<uint16_t, mask_w, 32>(
                            l1_write_addr_src1 + last_tile_idx_in_block * src1_tile_size,
                            static_cast<uint16_t>(BF16_ZERO));
                    }
                }

                cb_push_back(src0_cb_id, current_block_size);
                cb_push_back(src1_cb_id, current_block_size);
            }
        }
    }
}
