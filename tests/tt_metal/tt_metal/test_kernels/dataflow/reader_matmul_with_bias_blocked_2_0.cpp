// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t src0_addr = get_arg(args::src0_addr);
    uint32_t src0_dram_bank_id = get_arg(args::src0_dram_bank_id);
    uint32_t src1_addr = get_arg(args::src1_addr);
    uint32_t src1_dram_bank_id = get_arg(args::src1_dram_bank_id);
    uint32_t num_blocks = get_arg(args::num_blocks);

    uint32_t in0_block_tile_cnt = get_arg(args::in0_block_tile_cnt);
    uint32_t in1_block_tile_cnt = get_arg(args::in1_block_tile_cnt);
    uint32_t in0_block_size_bytes = get_arg(args::in0_block_size_bytes);
    uint32_t in1_block_size_bytes = get_arg(args::in1_block_size_bytes);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;

    for (uint32_t i = 0; i < num_blocks; i++) {
        dfb_in0.reserve_back(in0_block_tile_cnt);
        dfb_in1.reserve_back(in1_block_tile_cnt);

        noc.async_read(dram, dfb_in0, in0_block_size_bytes, {.bank_id = src0_dram_bank_id, .addr = src0_addr}, {});
        noc.async_read(dram, dfb_in1, in1_block_size_bytes, {.bank_id = src1_dram_bank_id, .addr = src1_addr}, {});

        noc.async_read_barrier();

        dfb_in0.push_back(in0_block_tile_cnt);
        dfb_in1.push_back(in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
