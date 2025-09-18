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
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    // Single tile size in bytes (FP16_B format)
    constexpr uint32_t single_tile_size = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    // Setup NOC for writing to DRAM
    const InterleavedAddrGenFast<false> dst_addrg = {
        .bank_base_address = dst_addr, .page_size = single_tile_size, .data_format = data_format};

    // Write output tiles to DRAM
    // The fused kernel produces reduced output tiles (fewer than input due to height reduction)

    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        // Wait for output tile to be available from compute kernel
        cb_wait_front(cb_id_out0, 1);

        // Get the tile from circular buffer and write to DRAM
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(tile_idx, dst_addrg, l1_read_addr);
        noc_async_write_barrier();

        // Mark tile as consumed
        cb_pop_front(cb_id_out0, 1);
    }
}
