// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

#include <cmath>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with masks
constexpr auto cb_mask_w_idx = tt::CBIndex::c_4;   // Mask for input inner dimension
constexpr auto cb_mask_hw_idx = tt::CBIndex::c_5;  // Mask for hidden inner dimension
// CBs with intermediate computations
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;        // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;        // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;          // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;  // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// TODO(maciek): Update write_cb_block_to_dram to allign Stas's change.
// TODO(maciek): Move all write_cb_block to common utils file, and reuse them in other
// operations.
// -----------------------------------------------------------------------------
// Shared utility: DO NOT CHANGE
inline void write_cb_block_to_dram(
    uint32_t cb_idx,
    const InterleavedAddrGenFast</* is dram */ true>& addr_gen,
    uint32_t start_idx,
    uint32_t current_block_size,
    uint32_t tile_bytes) {
    uint32_t l1_read_addr = get_read_ptr(cb_idx);

    // Wait for a full block in CB, but only write as many as are valid in the tail
    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        noc_async_write_tile(start_idx + block_idx, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
}
// -----------------------------------------------------------------------------

void kernel_main() {
    uint32_t ra = 0;
    uint32_t y_addr = get_arg_val<uint32_t>(ra++);  // DRAM base for Y
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_y_idx);
    const DataFormat data_fmt = get_dataformat(cb_y_idx);

    const InterleavedAddrGenFast<true> y_addr_gen = {
        .bank_base_address = y_addr, .page_size = tile_bytes, .data_format = data_fmt};

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

            // Wait for and write Y[r, c_block] - this becomes available after compute kernel finishes processing all
            // k_blocks for this (r, c_block) combination
            cb_wait_front(cb_y_idx, block_size);
            // Calculate starting tile index for Y. We write Y in row-major order, so the offset equals
            const uint32_t start_tile_idx = (r * Wt) + c_block_start;
            write_cb_block_to_dram(cb_y_idx, y_addr_gen, start_tile_idx, current_block_size, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_y_idx, block_size);
        }
    }
}
