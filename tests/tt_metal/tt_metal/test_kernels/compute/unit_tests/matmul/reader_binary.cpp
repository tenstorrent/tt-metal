// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);

    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_dram_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(in0_cb);
    uint32_t ublock_size_bytes_1 = get_tile_size(in1_cb);
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_dram_bank_id, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_dram_bank_id, src1_addr);

        cb_reserve_back(in0_cb, ublock_size_tiles);
        cb_reserve_back(in1_cb, ublock_size_tiles);

        l1_write_addr_in0 = get_write_ptr(in0_cb);
        l1_write_addr_in1 = get_write_ptr(in1_cb);

        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        noc_async_read_barrier();

        cb_push_back(in0_cb, ublock_size_tiles);
        cb_push_back(in1_cb, ublock_size_tiles);

        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }
}
