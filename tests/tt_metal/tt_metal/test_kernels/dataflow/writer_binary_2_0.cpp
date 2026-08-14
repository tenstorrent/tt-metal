// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

// Dual-output writer: drains two DFBs ("in0"/"in1" consumer accessors) to two
// DRAM buffers, one tile from each per iteration.
void kernel_main() {
    uint32_t dst0_addr = get_arg(args::dst0_addr);
    uint32_t dst0_bank_id = get_arg(args::dst0_bank_id);
    uint32_t dst1_addr = get_arg(args::dst1_addr);
    uint32_t dst1_bank_id = get_arg(args::dst1_bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    constexpr uint32_t ublock_size_tiles = 1;

    DataflowBuffer buff_out0(dfb::in0);
    DataflowBuffer buff_out1(dfb::in1);
    const uint32_t ublock_size_bytes_0 = buff_out0.get_entry_size() * ublock_size_tiles;
    const uint32_t ublock_size_bytes_1 = buff_out1.get_entry_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        buff_out0.wait_front(ublock_size_tiles);
        noc.async_write(
            buff_out0,
            AllocatorBank<AllocatorBankType::DRAM>{},
            ublock_size_bytes_0,
            {},
            {.bank_id = dst0_bank_id, .addr = dst0_addr});
        buff_out1.wait_front(ublock_size_tiles);
        noc.async_write(
            buff_out1,
            AllocatorBank<AllocatorBankType::DRAM>{},
            ublock_size_bytes_1,
            {},
            {.bank_id = dst1_bank_id, .addr = dst1_addr});
        noc.async_write_barrier();
        buff_out0.pop_front(ublock_size_tiles);
        buff_out1.pop_front(ublock_size_tiles);
        dst0_addr += ublock_size_bytes_0;
        dst1_addr += ublock_size_bytes_1;
    }
}
