// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst0_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t cb_id_out0 = get_arg_val<uint32_t>(3);
    uint32_t dst1_addr  = get_arg_val<uint32_t>(4);
    uint32_t dst1_noc_x = get_arg_val<uint32_t>(5);
    uint32_t dst1_noc_y = get_arg_val<uint32_t>(6);
    uint32_t cb_id_out1 = get_arg_val<uint32_t>(7);
    uint32_t num_tiles = get_arg_val<uint32_t>(8);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(9);

    uint32_t ublock0_size_bytes = get_tile_size(cb_id_out0) * ublock_size_tiles;
    uint32_t ublock1_size_bytes = get_tile_size(cb_id_out1) * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst0_noc_addr = get_noc_addr(dst0_noc_x, dst0_noc_y, dst0_addr);
        uint64_t dst1_noc_addr = get_noc_addr(dst1_noc_x, dst1_noc_y, dst1_addr);

        cb_wait_front(cb_id_out0, ublock_size_tiles);
        cb_wait_front(cb_id_out1, ublock_size_tiles);
        uint32_t l1_read_addr0 = get_read_ptr(cb_id_out0);
        uint32_t l1_read_addr1 = get_read_ptr(cb_id_out1);

        noc_async_write(l1_read_addr0, dst0_noc_addr, ublock0_size_bytes);
        noc_async_write(l1_read_addr1, dst1_noc_addr, ublock1_size_bytes);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);
        cb_pop_front(cb_id_out1, ublock_size_tiles);
        dst0_addr += ublock0_size_bytes;
        dst1_addr += ublock1_size_bytes;
    }
}
