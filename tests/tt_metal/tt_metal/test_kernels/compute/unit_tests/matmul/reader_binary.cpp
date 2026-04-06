// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"

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

    experimental::Noc noc;
    experimental::CircularBuffer cb0(in0_cb);
    experimental::CircularBuffer cb1(in1_cb);
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src0;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src1;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb0.reserve_back(ublock_size_tiles);
        cb1.reserve_back(ublock_size_tiles);

        noc.async_read(dram_src0, cb0, ublock_size_bytes_0, {.bank_id = src0_dram_bank_id, .addr = src0_addr}, {});
        noc.async_read(dram_src1, cb1, ublock_size_bytes_1, {.bank_id = src1_dram_bank_id, .addr = src1_addr}, {});

        noc.async_read_barrier();

        cb0.push_back(ublock_size_tiles);
        cb1.push_back(ublock_size_tiles);

        src0_addr += ublock_size_bytes_0;
        src1_addr += ublock_size_bytes_1;
    }
}
