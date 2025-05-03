// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "remote_circular_buffer_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

constexpr uint32_t noc = get_compile_time_arg_val(0);
constexpr uint32_t num_receivers = get_compile_time_arg_val(1);
constexpr uint32_t num_layers = get_compile_time_arg_val(2);
constexpr uint32_t remote_cb_id = get_compile_time_arg_val(3);

tt_l1_ptr uint32_t* noc_x;
tt_l1_ptr uint32_t* noc_y;
tt_l1_ptr uint32_t* coalesced_page_size;
tt_l1_ptr uint32_t* coalesced_num_pages;
tt_l1_ptr uint32_t* num_blocks;
tt_l1_ptr uint32_t* block_num_tiles;
tt_l1_ptr uint32_t* page_size;
tt_l1_ptr uint32_t* num_tile_rows;

uint32_t start_page_size;
uint32_t layer = 0;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));
    noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_receivers)));

    coalesced_page_size = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    coalesced_num_pages = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    num_blocks = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    block_num_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    page_size = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    num_tile_rows = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));

    start_page_size = page_size[0];

    constexpr uint32_t cb_id = 0;

    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t curr_coalesced_page_size = coalesced_page_size[l];
        uint32_t curr_coalesced_num_pages = coalesced_num_pages[l];
        uint32_t curr_num_blocks = num_blocks[l];
        uint32_t curr_block_num_tiles = block_num_tiles[l];
        uint32_t curr_page_size = page_size[l];
        uint32_t curr_num_tile_rows = num_tile_rows[l];
        uint32_t curr_receiver_block_num_tiles = curr_block_num_tiles / num_receivers;

        uint32_t curr_block_size = curr_receiver_block_num_tiles * curr_page_size;
        experimental::resize_remote_sender_cb_interface(remote_cb_id, curr_block_size, noc_index);

        for (uint32_t block = 0; block < curr_num_blocks; ++block) {
            cb_wait_front(cb_id, curr_block_num_tiles);

            uint32_t local_cb_addr = get_read_ptr(cb_id);
            experimental::remote_cb_reserve_back(remote_cb_id, 1);
            experimental::remote_cb_push_back_and_write_pages(
                remote_cb_id,
                local_cb_addr,
                1,
                curr_num_tile_rows,
                curr_coalesced_num_pages,
                curr_coalesced_page_size,
                noc);

            cb_pop_front(cb_id, curr_block_num_tiles);
        }
        layer++;
    }
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
}
