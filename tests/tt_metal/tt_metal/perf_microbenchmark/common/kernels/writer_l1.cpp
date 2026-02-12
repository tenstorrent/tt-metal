// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "experimental/circular_buffer.h"
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

    experimental::Noc noc_if{noc};
    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};

    constexpr uint32_t local_cb_id = 0;
    experimental::CircularBuffer local_cb{local_cb_id};

    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t curr_coalesced_page_size = coalesced_page_size[l];
        uint32_t curr_coalesced_num_pages = coalesced_num_pages[l];
        uint32_t curr_num_blocks = num_blocks[l];
        uint32_t curr_block_num_tiles = block_num_tiles[l];
        uint32_t curr_page_size = page_size[l];
        uint32_t curr_num_tile_rows = num_tile_rows[l];
        uint32_t curr_receiver_block_num_tiles = curr_block_num_tiles / num_receivers;

        uint32_t curr_block_size = curr_receiver_block_num_tiles * curr_page_size;
        remote_cb.set_receiver_page_size(noc_if, curr_block_size);

        for (uint32_t block = 0; block < curr_num_blocks; ++block) {
            local_cb.wait_front(curr_block_num_tiles);

            remote_cb.reserve_back(1);
            remote_cb.push_back(
                noc_if, local_cb, 1, curr_num_tile_rows, curr_coalesced_num_pages, curr_coalesced_page_size);

            local_cb.pop_front(curr_block_num_tiles);
        }
        layer++;
    }
    remote_cb.commit();
}
