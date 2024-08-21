// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    constexpr std::uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);
    std::uint32_t src_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr bool IS_DRAM = false;
    const uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = get_tile_size(cb_id);
    InterleavedAddrGen<IS_DRAM> src_addrgen = {
        .bank_base_address = src_addr_base,
        .page_size = page_size,
    };

    // read tiles from src to CB
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr(i, src_addrgen);

        cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id, ublock_size_tiles);
    }
}
