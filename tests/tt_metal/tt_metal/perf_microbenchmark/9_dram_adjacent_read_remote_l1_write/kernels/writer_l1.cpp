// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"


void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t noc = get_compile_time_arg_val(4);

    const uint32_t vc = get_arg_val<uint32_t>(0);
    const uint32_t noc_x = get_arg_val<uint32_t>(1);
    const uint32_t noc_y = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = 0;

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    const uint64_t l1_noc_write_addr = get_noc_addr(noc_x, noc_y, l1_write_addr, noc);

    noc_async_write_one_packet_set_state(l1_noc_write_addr, page_size, noc, vc);

    for (uint32_t block = 0; block < num_blocks; ++block) {

        auto remote_l1_write_addr = l1_noc_write_addr;

        cb_wait_front(cb_id, block_num_tiles);
        auto l1_read_addr = get_read_ptr(cb_id);

        for (uint32_t h = 0; h < num_pages; ++h) {
            noc_async_write_one_packet_with_state(l1_read_addr, remote_l1_write_addr, noc);
            l1_read_addr += page_size;
            remote_l1_write_addr += page_size;
        }

        noc_async_write_barrier(noc);

        cb_pop_front(cb_id, block_num_tiles);

    }


}
