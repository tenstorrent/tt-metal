// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

void kernel_main() {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);

    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_dram_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_blocks = get_arg_val<uint32_t>(4);
    uint32_t in0_block_tile_cnt = get_arg_val<uint32_t>(5);
    uint32_t in1_block_tile_cnt = get_arg_val<uint32_t>(6);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(7);
    uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(8);

    experimental::CircularBuffer cb0(in0_cb);
    experimental::CircularBuffer cb1(in1_cb);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src0;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_dram_bank_id, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_dram_bank_id, src1_addr);

        cb0.reserve_back(in0_block_tile_cnt);
        cb1.reserve_back(in1_block_tile_cnt);

        noc.async_read(dram_src0, cb0, in0_block_size_bytes, {.bank_id = src0_dram_bank_id, .addr = src0_addr}, {});
        noc.async_read(dram_src1, cb1, in1_block_size_bytes, {.bank_id = src1_dram_bank_id, .addr = src1_addr}, {});

        noc.async_read_barrier();

        cb0.push_back(in0_block_tile_cnt);
        cb1.push_back(in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
