// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"

inline __attribute__((always_inline)) void pop_from_cb_and_write(
    experimental::CircularBuffer& cb,
    uint32_t num_tiles_per_cb,
    uint32_t ublock_size_tiles,
    uint32_t ublock_size_bytes,
    uint32_t bank_id,
    uint32_t& dram_buffer_dst_addr) {
    experimental::Noc noc;
    for (uint32_t i = 0; i < num_tiles_per_cb; i += ublock_size_tiles) {
        // DRAM NOC dst address
        std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dram_buffer_dst_addr);

        cb.wait_front(ublock_size_tiles);
        noc.async_write(
            cb,
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dram_buffer_dst_addr});
        noc.async_write_barrier();
        cb.pop_front(ublock_size_tiles);
        dram_buffer_dst_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t bank_id               = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles_per_cb      = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t ublock_size_tiles = get_compile_time_arg_val(1);

    experimental::CircularBuffer cb(cb_id);
    uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

    pop_from_cb_and_write(cb, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes, bank_id, dram_buffer_dst_addr);
}
