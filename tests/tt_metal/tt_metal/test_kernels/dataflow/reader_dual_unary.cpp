// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

inline void read_tiles(
    experimental::Noc& noc, uint32_t num_tiles, uint32_t src_addr, uint32_t bank_id, uint32_t cb_id_in) {
    // ublocks size defined in tiles
    experimental::CircularBuffer cb(cb_id_in);
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.reserve_back(ublock_size_tiles);
        noc.async_read(dram_bank, cb, ublock_size_bytes, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        cb.push_back(ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    uint32_t src_addr_0 = get_arg_val<uint32_t>(0);
    uint32_t bank_id_0 = get_arg_val<uint32_t>(1);
    uint32_t src_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t bank_id_1 = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    experimental::Noc noc;

    read_tiles(noc, num_tiles, src_addr_0, bank_id_0, cb_id_in0);
    read_tiles(noc, num_tiles, src_addr_1, bank_id_1, cb_id_in1);
}
