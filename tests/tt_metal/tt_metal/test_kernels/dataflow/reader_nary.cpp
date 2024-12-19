// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    uint32_t num_reads = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_read = get_arg_val<uint32_t>(1);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_reads; i++) {
        uint32_t src_addr = get_arg_val<uint32_t>(2 + i * 4);
        uint32_t src_noc_x = get_arg_val<uint32_t>(3 + i * 4);
        uint32_t src_noc_y = get_arg_val<uint32_t>(4 + i * 4);
        uint32_t cb_id_in = get_arg_val<uint32_t>(5 + i * 4);

        uint32_t ublock_size_bytes = get_tile_size(cb_id_in);

        // read a ublock of tiles from src to CB, and then push the ublock to unpacker
        for (uint32_t i = 0; i < num_tiles_per_read; i += ublock_size_tiles) {
            uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);

            cb_reserve_back(cb_id_in, ublock_size_tiles);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

            noc_async_read_barrier();

            cb_push_back(cb_id_in, ublock_size_tiles);
            src_addr += ublock_size_bytes;
        }
    }
}
