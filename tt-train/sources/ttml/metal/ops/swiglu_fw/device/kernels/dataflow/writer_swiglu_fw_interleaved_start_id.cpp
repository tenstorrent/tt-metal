// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with intermediate computations
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial (X @ W1)[r, k_block] between p_blocks
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial (X @ W3)[r, k_block] between p_blocks
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t ra = 0;
    uint32_t y_addr = get_arg_val<uint32_t>(ra++);  // DRAM base for Y
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_y_idx);

    // Address generator
    constexpr auto y_args = TensorAccessorArgs<2>();
    const auto y_address_generator = TensorAccessor(y_args, y_addr, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;

    // ================== Loop structure matches compute kernel ==================
    // for r in rows:
    //   for c_block in c_blocks:
    //     [compute processes all c for this (r, c_block)]
    //     write Y[r, c_block] to DRAM
    // ============================================================================
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Loop over output columns in blocks - matches compute kernel order
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t current_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);
            // Calculate starting tile index for Y. We write Y in row-major order, so the offset equals
            const uint32_t start_tile_idx = (r * Wt) + c_block_start;
            // Wait for and write Y[r, c_block] - this becomes available after compute kernel finishes processing all
            // k_blocks for this (r, c_block) combination
            write_tiles_by_row(
                cb_y_idx, y_address_generator, start_tile_idx, current_block_size, tile_bytes, block_size);
        }
    }
}
