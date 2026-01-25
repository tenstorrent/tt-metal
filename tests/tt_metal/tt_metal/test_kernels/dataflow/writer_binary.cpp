// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t dst0_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst0_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t cb_id_out0 = get_arg_val<uint32_t>(2);
    uint32_t dst1_addr = get_arg_val<uint32_t>(3);
    uint32_t dst1_dram_bank_id = get_arg_val<uint32_t>(4);
    uint32_t cb_id_out1 = get_arg_val<uint32_t>(5);
    uint32_t num_tiles = get_arg_val<uint32_t>(6);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(7);

    experimental::CircularBuffer cb0(cb_id_out0);
    experimental::CircularBuffer cb1(cb_id_out1);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_dst;

    uint32_t ublock0_size_bytes = cb0.get_tile_size() * ublock_size_tiles;
    uint32_t ublock1_size_bytes = cb1.get_tile_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb0.wait_front(ublock_size_tiles);
        cb1.wait_front(ublock_size_tiles);

        noc.async_write(cb0, dram_dst, ublock0_size_bytes, {}, {.bank_id = dst0_dram_bank_id, .addr = dst0_addr});
        noc.async_write(cb1, dram_dst, ublock1_size_bytes, {}, {.bank_id = dst1_dram_bank_id, .addr = dst1_addr});

        noc.async_write_barrier();

        cb0.pop_front(ublock_size_tiles);
        cb1.pop_front(ublock_size_tiles);
        dst0_addr += ublock0_size_bytes;
        dst1_addr += ublock1_size_bytes;
    }
}
