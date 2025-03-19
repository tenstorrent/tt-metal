// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    std::uint32_t buffer_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(2);
    std::uint32_t num_repetitions = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);
    uint32_t block_size_bytes = get_tile_size(cb_id) * block_size_tiles;

    for (uint32_t j = 0; j < num_repetitions; j++) {
        uint32_t src_addr = buffer_src_addr;
        for (uint32_t i = 0; i < num_tiles; i += block_size_tiles) {
            std::uint64_t buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);
            cb_reserve_back(cb_id, block_size_tiles);

            if (j == 0) {
                uint32_t l1_write_addr = get_write_ptr(cb_id);
                noc_async_read(buffer_src_noc_addr, l1_write_addr, block_size_bytes);
                noc_async_read_barrier();
            }

            cb_push_back(cb_id, block_size_tiles);
            src_addr += block_size_bytes;
        }
    }
}
