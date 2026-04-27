// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    experimental::Noc noc;
    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes;

#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(0);
    experimental::DataflowBuffer buff_in(dfb_in_id);
    ublock_size_bytes = buff_in.get_entry_size() * ublock_size_tiles;
#else
    constexpr uint32_t cb_id_in0 = 0;
    experimental::CircularBuffer buff_in(cb_id_in0);
    ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;
#endif
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        buff_in.reserve_back(ublock_size_tiles);
        noc.async_read(
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            buff_in,
            ublock_size_bytes,
            {.bank_id = bank_id, .addr = src_addr},
            {});
        noc.async_read_barrier();
        buff_in.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
