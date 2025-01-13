// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages_w_per_receiver = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles_h = get_compile_time_arg_val(2);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t noc = get_compile_time_arg_val(5);

    const uint32_t vc = get_arg_val<uint32_t>(0);
    // First L1 writer core coordinates
    const uint32_t noc_x1 = get_arg_val<uint32_t>(1);
    const uint32_t noc_y1 = get_arg_val<uint32_t>(2);
    // Second L1 writer core coordinates
    const uint32_t noc_x2 = get_arg_val<uint32_t>(3);
    const uint32_t noc_y2 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = 0;

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    const uint64_t l1_noc_write_addr1 = get_noc_addr(noc_x1, noc_y1, l1_write_addr, noc);
    const uint64_t l1_noc_write_addr2 = get_noc_addr(noc_x2, noc_y2, l1_write_addr, noc);

    for (uint32_t block = 0; block < num_blocks; ++block) {  // Iterate over blocks

        cb_wait_front(cb_id, block_num_tiles);

        for (uint32_t core_id = 0; core_id < 2; ++core_id) {  // Iterate over two neighboring cores
            uint64_t l1_noc_write_addr_for_receiver_core = 0;
            uint32_t l1_read_addr = get_read_ptr(cb_id);
            if (core_id == 0) {
                l1_noc_write_addr_for_receiver_core =
                    l1_noc_write_addr1;  // Set write pointer to start of cb for first core
            } else {
                l1_noc_write_addr_for_receiver_core =
                    l1_noc_write_addr2;  // Set write pointer to start of cb for second core
                l1_read_addr += page_size * num_pages_w_per_receiver;  // Stride read pointer to start of second core
            }

            noc_async_write_one_packet_set_state(
                l1_noc_write_addr_for_receiver_core, page_size, noc, vc);  // Set state to write a page to noc/vc

            for (uint32_t h = 0; h < num_tiles_h; ++h) {  // Iterate over page rows per receiver core
                for (uint32_t w = 0; w < num_pages_w_per_receiver;
                     ++w) {  // Iterate over page columns per receiver core
                    noc_async_write_one_packet_with_state(l1_read_addr, l1_noc_write_addr_for_receiver_core, noc);
                    l1_read_addr += page_size;
                    l1_noc_write_addr_for_receiver_core += page_size;
                }
                l1_read_addr += page_size * num_pages_w_per_receiver;  // Stride read pointer over other core's data
            }
        }

        noc_async_write_barrier(noc);

        cb_pop_front(cb_id, block_num_tiles);
    }
}
