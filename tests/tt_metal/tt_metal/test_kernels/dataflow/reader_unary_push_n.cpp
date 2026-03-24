// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif
#include "experimental/endpoints.h"
#include "experimental/noc.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
#ifdef ARCH_QUASAR
    uint32_t dfb_id_in0 = get_arg_val<uint32_t>(3);
#else
    uint32_t cb_id_in0 = get_arg_val<uint32_t>(3);
#endif
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(4);
    bool reader_only = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb(dfb_id_in0);
    uint32_t ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;
#else
    experimental::CircularBuffer cb(cb_id_in0);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
#endif

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#ifdef ARCH_QUASAR
        if (reader_only == false) {
            dfb.reserve_back(ublock_size_tiles);
        }
        noc.async_read(dram_src, dfb, ublock_size_bytes, {.bank_id = src_dram_bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        if (reader_only == false) {
            dfb.push_back(ublock_size_tiles);
        }
#else
        if (reader_only == false) {
            cb.reserve_back(ublock_size_tiles);
        }
        noc.async_read(dram_src, cb, ublock_size_bytes, {.bank_id = src_dram_bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        if (reader_only == false) {
            cb.push_back(ublock_size_tiles);
        }
#endif
        src_addr += ublock_size_bytes;
    }
}
