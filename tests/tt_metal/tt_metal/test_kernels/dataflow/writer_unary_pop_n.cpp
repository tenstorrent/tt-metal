// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t dst_dram_bank_id = get_arg(args::dst_dram_bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t ublock_size_tiles = get_arg(args::ublock_size_tiles);
    bool writer_only = get_arg(args::writer_only);

    DataflowBuffer dfb(dfb::in);
    uint32_t ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (writer_only == false) {
            dfb.wait_front(ublock_size_tiles);
        }
        noc.async_write(dfb, dram_dst, ublock_size_bytes, {}, {.bank_id = dst_dram_bank_id, .addr = dst_addr});

        noc.async_write_barrier();
        if (writer_only == false) {
            dfb.pop_front(ublock_size_tiles);
        }
        dst_addr += ublock_size_bytes;
    }
}
