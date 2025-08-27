// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CB with output Y data from compute kernel
constexpr auto cb_y_idx = tt::CBIndex::c_7;  // matches compute kernel's final output CB

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// -----------------------------------------------------------------------------
// Shared utility: DO NOT CHANGE
inline void write_cb_block_to_dram(
    uint32_t cb_idx,
    const InterleavedAddrGenFast</* is dram */ true>& addr_gen,
    uint32_t start_idx,
    uint32_t block_size,
    uint32_t current_block_size,
    uint32_t tile_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);

    // Wait for a full block in CB, but only write as many as are valid in the tail
    for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
        noc_async_write_tile(start_idx + block_idx, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
}
// -----------------------------------------------------------------------------

void kernel_main() {
    return;
    /*
    uint32_t ra = 0;
    uint32_t y_addr = get_arg_val<uint32_t>(ra++);  // DRAM base for Y
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_y_idx);
    const DataFormat data_fmt = get_dataformat(cb_y_idx);

    const InterleavedAddrGenFast<true> y_addr_gen = {
        .bank_base_address = y_addr, .page_size = tile_bytes, .data_format = data_fmt};

    const uint32_t end_row = start_row + num_rows_to_process;

    for (uint32_t r = start_row; r < end_row; ++r) {
        // Loop over output columns in blocks, tail-safe
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t current_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);

            const uint32_t start_tile_idx = (r * Wt) + c_block_start;

            // Write from CB to DRAM (helper waits for block_size, writes only current_block_size)
            write_cb_block_to_dram(cb_y_idx, y_addr_gen, start_tile_idx, block_size, current_block_size, tile_bytes);

            noc_async_write_barrier();
            cb_pop_front(cb_y_idx, block_size);  // pop the full block, even if tail was smaller
        }
    }
    */
}
