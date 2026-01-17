// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
#if INTERFACE_WITH_L1 == 1
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::L1;
#else
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
#endif

    experimental::Noc noc(noc_index);
    experimental::CircularBuffer cb(cb_id);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.reserve_back(ublock_size_tiles);
        noc.async_read(
            experimental::AllocatorBank<bank_type>(),
            cb,
            ublock_size_bytes,
            {.bank_id = src_bank_id, .addr = src_addr},
            {});
        noc.async_read_barrier();
        cb.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
