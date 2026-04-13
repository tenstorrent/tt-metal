// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

#ifdef ARCH_QUASAR
    uint32_t dfb_id = get_compile_time_arg_val(0);
    experimental::DataflowBuffer dfb_in(dfb_id);
#else
    constexpr uint32_t cb_id_in0 = 0;
#endif

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
#ifdef ARCH_QUASAR
    uint32_t ublock_size_bytes = dfb_in.get_entry_size() * ublock_size_tiles;
    experimental::Noc noc;
#else
    uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;
#endif

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#ifdef ARCH_QUASAR
        dfb_in.reserve_back(ublock_size_tiles);
        noc.async_read(
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            dfb_in,
            ublock_size_bytes,
            {.bank_id = bank_id, .addr = src_addr},
            {});
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_tiles);
#else
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
#endif
        src_addr += ublock_size_bytes;
    }
}
