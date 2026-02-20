// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(2);  // ones vector for matmul reduction
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(4);  // block size - must match compute/writer kernels
    constexpr uint32_t mask_w = get_compile_time_arg_val(5);           // padding mask width (0 if no padding)
    // Set up tensor accessors
    constexpr auto softmax_output_args = TensorAccessorArgs<6>();
    constexpr auto upstream_grad_args = TensorAccessorArgs<7>();

    // Common runtime args (shared across all cores)
    const uint32_t softmax_output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t upstream_grad_addr = get_common_arg_val<uint32_t>(1);

    // Per-core runtime args: work assignment for this core (supports arbitrary CoreRangeSet)
    const uint32_t start_row = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

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

                const uint32_t block_start_tile = row_start_tile + block_start;

                // CB API requires the same tile count in all cb_* calls (must divide CB size).
                // Always reserve/push tiles_per_block; only current_block_size tiles are read from DRAM.
                read_tiles_by_row</* UseBarrier = */ false>(
                    src0_cb_id,
                    softmax_output_accessor,
                    block_start_tile,
                    current_block_size,
                    src0_tile_size,
                    tiles_per_block);
                read_tiles_by_row</* UseBarrier = */ false>(
                    src1_cb_id,
                    upstream_grad_accessor,
                    block_start_tile,
                    current_block_size,
                    src1_tile_size,
                    tiles_per_block);

                noc_async_read_barrier();

                if constexpr (mask_w > 0) {
                    const uint32_t global_last_tile_idx = block_start + current_block_size - 1;
                    if (global_last_tile_idx == num_tiles_per_row - 1) {
                        const uint32_t last_tile_idx_in_block = current_block_size - 1;
                        const uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);
                        const uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);
                        fill_pad_tile<uint16_t, mask_w, 32>(
                            l1_write_addr_src0 + last_tile_idx_in_block * src0_tile_size,
                            static_cast<uint16_t>(BF16_ZERO_BITS));
                        fill_pad_tile<uint16_t, mask_w, 32>(
                            l1_write_addr_src1 + last_tile_idx_in_block * src1_tile_size,
                            static_cast<uint16_t>(BF16_ZERO_BITS));
                    }
                }

                // Zero-fill tail padding so compute always sees tiles_per_block tiles (no padding logic in compute).
                if (current_block_size < tiles_per_block) {
                    const uint32_t pad_slots = tiles_per_block - current_block_size;
                    fill_reserved_tiles_with_zero(src0_cb_id, current_block_size, pad_slots, src0_tile_size);
                    fill_reserved_tiles_with_zero(src1_cb_id, current_block_size, pad_slots, src1_tile_size);
                }

                cb_push_back(src0_cb_id, tiles_per_block);
                cb_push_back(src1_cb_id, tiles_per_block);
            }
        }
    }
}
