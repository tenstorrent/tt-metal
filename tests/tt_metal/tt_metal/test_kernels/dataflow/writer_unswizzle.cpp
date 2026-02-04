// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

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

    experimental::CircularBuffer cb(cb_id_out0);
    experimental::Noc noc;

    for (uint32_t sb_m = 0; sb_m < num_sub_blocks_m; sb_m++) {
        uint32_t dram_address_block_beginning = dram_address_block_row_beginning;
        for (uint32_t sb_n = 0; sb_n < num_sub_blocks_n; sb_n++) {
            uint32_t dram_address_r = dram_address_block_beginning;
            for (uint32_t r = 0; r < inner_r; r++) {
                uint32_t dram_address_c = dram_address_r;
                for(uint32_t c = 0; c < inner_c; c++) {
                    cb.wait_front(ublock_size_tiles);

                    noc.async_write(
                        cb,
                        experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                        ublock_size_bytes,
                        {},
                        {.bank_id = dst_bank_id, .addr = dram_address_c});

                    noc.async_write_barrier();

                    cb.pop_front(ublock_size_tiles);
                    dram_address_c += ublock_size_bytes;
                }
                dram_address_r += stride_r;  // goto next row within sub-block
            }
            dram_address_block_beginning += stride_subblock_c;  // move to next sub-block on c dim
        }
        dram_address_block_row_beginning += stride_subblock_r;  // move to next sub-block on r dim
    }
}
