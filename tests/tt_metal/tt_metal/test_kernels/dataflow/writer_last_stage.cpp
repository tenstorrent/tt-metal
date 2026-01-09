// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
// #include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    std::uint32_t buffer_dst_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(2);
    std::uint32_t num_repetitions = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_tiles = get_compile_time_arg_val(1);

    experimental::CircularBuffer cb(cb_id);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_dst;

    uint32_t block_size_bytes = cb.get_tile_size() * block_size_tiles;

    for (uint32_t j = 0; j < num_repetitions; j++) {
        uint32_t dst_addr = buffer_dst_addr;
        for (uint32_t i = 0; i < num_tiles; i += block_size_tiles) {
            cb.wait_front(block_size_tiles);

            if (j == 0) {
                uint32_t l1_read_addr = get_read_ptr(cb_id);
                noc.async_write(cb, dram_dst, block_size_bytes, {}, {.bank_id = dst_bank_id, .addr = dst_addr});
                noc.async_write_barrier();

                // some delay to test backpressure
                // volatile uint32_t *l1_read_addr_ptr = reinterpret_cast<volatile tt_l1_ptr
                // uint32_t*>(BRISC_BREAKPOINT); for (int delay = 0; delay < 10000; delay++) {
                //     *l1_read_addr_ptr = 1;
                // }
            }

            cb.pop_front(block_size_tiles);
            dst_addr += block_size_bytes;
        }
    }
}
