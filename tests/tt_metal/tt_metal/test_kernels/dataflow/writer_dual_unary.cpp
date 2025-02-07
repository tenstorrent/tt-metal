// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

inline void write_tiles(uint32_t num_tiles, uint32_t dst_addr, uint32_t bank_id, uint32_t cb_id_out) {
    // single-tile ublocks
    uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);

        cb_wait_front(cb_id_out, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    uint32_t dst_addr_0 = get_arg_val<uint32_t>(0);
    uint32_t bank_id_0 = get_arg_val<uint32_t>(1);
    uint32_t dst_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t bank_id_1 = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_out1 = tt::CBIndex::c_17;

    write_tiles(num_tiles, dst_addr_0, bank_id_0, cb_id_out0);
    write_tiles(num_tiles, dst_addr_1, bank_id_1, cb_id_out1);
}
