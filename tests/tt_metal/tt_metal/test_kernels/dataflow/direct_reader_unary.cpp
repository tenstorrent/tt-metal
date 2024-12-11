// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
#if INTERFACE_WITH_L1 == 1
    constexpr bool read_from_dram = false;
#else
    constexpr bool read_from_dram = true;
#endif
    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id) * ublock_size_tiles;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_buffer_noc_addr = get_noc_addr_from_bank_id<read_from_dram>(src_bank_id, src_addr);

        cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(src_buffer_noc_addr, l1_write_addr, ublock_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }
}
