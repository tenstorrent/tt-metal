// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t bank_id = get_arg(args::bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    constexpr uint32_t ublock_size_tiles = 1;

    DataflowBuffer buff_out(dfb::in);
    const uint32_t ublock_size_bytes = buff_out.get_entry_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        buff_out.wait_front(ublock_size_tiles);
        noc.async_write(
            buff_out,
            AllocatorBank<AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        buff_out.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
