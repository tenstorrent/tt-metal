// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

#if INTERFACE_WITH_L1 == 1
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::L1;
#else
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
#endif

    experimental::Noc noc(noc_index);
    experimental::CircularBuffer cb(cb_id);
    // single-tile ublocks
    uint32_t ublock_size_bytes = cb.get_tile_size();
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.wait_front(ublock_size_tiles);
        noc.async_write(
            cb,
            experimental::AllocatorBank<bank_type>(),
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
