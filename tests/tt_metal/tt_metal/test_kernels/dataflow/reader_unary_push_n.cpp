// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t cb_id_in0 = get_arg_val<uint32_t>(3);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(4);
    bool reader_only = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src;
    experimental::CircularBuffer cb(cb_id_in0);

    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_dram_bank_id, src_addr);
        if (reader_only == false) {
            cb.reserve_back(ublock_size_tiles);
        }
        noc.async_read(dram_src, cb, ublock_size_bytes, {.bank_id = src_dram_bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        if (reader_only == false) {
            cb.push_back(ublock_size_tiles);
        }
        src_addr += ublock_size_bytes;
    }
}
