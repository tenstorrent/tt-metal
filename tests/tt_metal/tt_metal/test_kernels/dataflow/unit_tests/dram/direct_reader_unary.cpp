// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#endif

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(cb_id);
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
    experimental::AllocatorBank<bank_type> src_dram;
    experimental::Noc noc;

    uint32_t ublock_size_bytes = dfb.get_entry_size();

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        dfb.reserve_back(ublock_size_tiles);
        noc.async_read(src_dram, dfb, ublock_size_bytes, {.bank_id = src_bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        dfb.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }

#else
    // ublocks size defined in tiles
    uint32_t ublock_size_bytes = get_tile_size(cb_id) * ublock_size_tiles;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i<num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);

        cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
#endif
}
