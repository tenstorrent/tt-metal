// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

inline void read_tiles(uint32_t num_tiles, uint32_t src_addr, uint32_t bank_id, uint32_t cb_id_in) {
    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in) * ublock_size_tiles;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

        cb_reserve_back(cb_id_in, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id_in, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    uint32_t src_addr_0 = get_arg_val<uint32_t>(0);
    uint32_t bank_id_0 = get_arg_val<uint32_t>(1);
    uint32_t src_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t bank_id_1 = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    read_tiles(num_tiles, src_addr_0, bank_id_0, cb_id_in0);
    read_tiles(num_tiles, src_addr_1, bank_id_1, cb_id_in1);
}
