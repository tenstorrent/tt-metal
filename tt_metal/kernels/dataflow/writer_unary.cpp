// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    Noc noc;
    constexpr uint32_t ublock_size_tiles = 1;

    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    CircularBuffer buff_out(cb_out_id);
    const uint32_t ublock_size_bytes = get_tile_size(cb_out_id) * ublock_size_tiles;

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
