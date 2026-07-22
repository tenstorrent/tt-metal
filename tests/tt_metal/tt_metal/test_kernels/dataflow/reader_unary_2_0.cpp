// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t src_addr = get_arg(args::src_addr);
    uint32_t bank_id = get_arg(args::bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    constexpr uint32_t ublock_size_tiles = 1;

    DataflowBuffer buff_in(dfb::out);
    uint32_t ublock_size_bytes = buff_in.get_entry_size() * ublock_size_tiles;
    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        buff_in.reserve_back(ublock_size_tiles);
        noc.async_read(
            AllocatorBank<AllocatorBankType::DRAM>{},
            buff_in,
            ublock_size_bytes,
            {.bank_id = bank_id, .addr = src_addr},
            {});
        noc.async_read_barrier();
        buff_in.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
