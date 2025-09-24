// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_bank = get_arg_val<uint32_t>(1);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(2);

    // Circular buffer index for output
    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;

    // Single tile size in bytes (FP16_b format)
    constexpr uint32_t single_tile_size = 2 * 1024;

    // Create DRAM interface for destination
    const InterleavedAddrGenFast<false> d = {
        .bank_base_address = dst_addr, .page_size = single_tile_size, .data_format = DataFormat::Float16_b};

    // Write output tiles to DRAM
    // For the multiple tiles test, this should write exactly 1 tile (the reduced result)
    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        // Wait for tile to be available in output CB
        cb_wait_front(cb_id_out, 1);

        // Get the tile data from the circular buffer
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write tile to DRAM
        noc_async_write_tile(tile_idx, d, l1_read_addr);
        noc_async_write_barrier();

        // Pop the tile from the circular buffer
        cb_pop_front(cb_id_out, 1);
    }
}
