// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t ublock_size_tiles = 1;

    experimental::CircularBuffer cb(cb_id_out0);
    experimental::Noc noc;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.wait_front(ublock_size_tiles);
        noc.async_write(
            cb,
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
