// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id);
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
         uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);

        cb_wait_front(cb_id, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
