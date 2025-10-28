// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t input_addr = get_compile_time_arg_val(0);
    constexpr uint32_t input_start_tile_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;

    uint64_t src_base_addr = get_noc_addr_from_bank_id<true>(bank_id, input_addr);
    noc_async_read_one_packet_set_state<true>(src_base_addr, page_size, vc);
    uint32_t l1_read_addr = 0;

    constexpr uint32_t total_num_blocks_in_buffer = 3;
    constexpr uint32_t total_num_trid = 4;
    uint32_t num_free_blocks_in_buffer = total_num_blocks_in_buffer;
    uint32_t curr_block_trid = 1;
    uint32_t block_trid_to_wait = 1;

    cb_reserve_back(cb_id, block_num_tiles);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read_tile_dram_sharded_set_trid(curr_block_trid);

        for (uint32_t h = 0; h < num_pages; ++h) {
            noc_async_read_tile_dram_sharded_with_state_with_trid(
                src_base_addr, l1_read_addr, l1_write_addr, curr_block_trid);
            l1_read_addr += page_size;
            l1_write_addr += page_size;
        }

        if (num_free_blocks_in_buffer == 2) {
            noc_async_read_barrier_with_trid(block_trid_to_wait);
            // wait for next block trid
            block_trid_to_wait = block_trid_to_wait == 3 ? 1 : (block_trid_to_wait + 1);
        } else {
            num_free_blocks_in_buffer -= 1;
        }

        if (curr_block_trid == total_num_blocks_in_buffer) {
            curr_block_trid = 1;
        } else {
            curr_block_trid += 1;
        }
    }
    // last block to wait
    noc_async_read_barrier_with_trid(block_trid_to_wait);
}
