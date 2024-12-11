// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr           = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id        = get_arg_val<uint32_t>(1);
    uint32_t inner_r            = get_arg_val<uint32_t>(2);
    uint32_t inner_c            = get_arg_val<uint32_t>(3);
    uint32_t num_sub_blocks_m   = get_arg_val<uint32_t>(4);
    uint32_t num_sub_blocks_n   = get_arg_val<uint32_t>(5);
    uint32_t stride_r           = get_arg_val<uint32_t>(6);
    uint32_t stride_subblock_r  = get_arg_val<uint32_t>(7);
    uint32_t stride_subblock_c  = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    uint32_t dram_address_block_row_beginning = dst_addr;
    for (uint32_t sb_m = 0; sb_m < num_sub_blocks_m; sb_m++) {
        uint32_t dram_address_block_beginning = dram_address_block_row_beginning;
        for (uint32_t sb_n = 0; sb_n < num_sub_blocks_n; sb_n++) {
            uint32_t dram_address_r = dram_address_block_beginning;
            for (uint32_t r = 0; r < inner_r; r++) {
                uint32_t dram_address_c = dram_address_r;
                for(uint32_t c = 0; c < inner_c; c++) {
                    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dram_address_c);

                    cb_wait_front(cb_id_out0, ublock_size_tiles);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

                    noc_async_write_barrier();

                    cb_pop_front(cb_id_out0, ublock_size_tiles);
                    dram_address_c += ublock_size_bytes;
                }
                dram_address_r += stride_r;  // goto next row within sub-block
            }
            dram_address_block_beginning += stride_subblock_c;  // move to next sub-block on c dim
        }
        dram_address_block_row_beginning += stride_subblock_r;  // move to next sub-block on r dim
    }
}
