// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr  = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_blocks = get_arg_val<uint32_t>(4);
    uint32_t in0_block_tile_cnt  = get_arg_val<uint32_t>(5);
    uint32_t in1_block_tile_cnt  = get_arg_val<uint32_t>(6);
    uint32_t in0_block_size_bytes  = get_arg_val<uint32_t>(7);
    uint32_t in1_block_size_bytes  = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);
    experimental::Noc noc;

    for(uint32_t i = 0; i < num_blocks; i++) {
        cb_in0.reserve_back(in0_block_tile_cnt);
        cb_in1.reserve_back(in1_block_tile_cnt);
        noc.async_read(
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            cb_in0,
            in0_block_size_bytes,
            {.bank_id = src0_bank_id, .addr = src0_addr},
            {});
        noc.async_read(
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            cb_in1,
            in1_block_size_bytes,
            {.bank_id = src1_bank_id, .addr = src1_addr},
            {});
        noc.async_read_barrier();
        cb_in0.push_back(in0_block_tile_cnt);
        cb_in1.push_back(in1_block_tile_cnt);
        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
