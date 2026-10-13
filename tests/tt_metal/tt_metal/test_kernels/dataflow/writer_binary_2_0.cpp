// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-output counterpart of writer_unary_2_0.cpp: drains DFBs out0 and out1, whose entries may
// differ in size (e.g. a values and an indices tile of different formats), into two DRAM buffers.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst0_addr = get_arg(args::dst0_addr);
    uint32_t dst0_bank_id = get_arg(args::dst0_bank_id);
    uint32_t dst1_addr = get_arg(args::dst1_addr);
    uint32_t dst1_bank_id = get_arg(args::dst1_bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb0(dfb::out0);
    DataflowBuffer dfb1(dfb::out1);
    const uint32_t tile_bytes_0 = dfb0.get_entry_size();
    const uint32_t tile_bytes_1 = dfb1.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb0.wait_front(1);
        dfb1.wait_front(1);
        noc.async_write(
            dfb0,
            AllocatorBank<AllocatorBankType::DRAM>{},
            tile_bytes_0,
            {},
            {.bank_id = dst0_bank_id, .addr = dst0_addr});
        noc.async_write(
            dfb1,
            AllocatorBank<AllocatorBankType::DRAM>{},
            tile_bytes_1,
            {},
            {.bank_id = dst1_bank_id, .addr = dst1_addr});
        noc.async_write_barrier();
        dfb0.pop_front(1);
        dfb1.pop_front(1);
        dst0_addr += tile_bytes_0;
        dst1_addr += tile_bytes_1;
    }
}
