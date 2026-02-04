// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "experimental/noc.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

constexpr uint32_t num_layers = get_compile_time_arg_val(0);
constexpr uint32_t remote_cb_id = get_compile_time_arg_val(1);

uint32_t rt_args_idx = 0;
uint32_t vc;
uint32_t noc_x;
uint32_t noc_y;
uint32_t receiver_index;
tt_l1_ptr uint32_t* page_size;
tt_l1_ptr uint32_t* num_blocks;
tt_l1_ptr uint32_t* block_num_tiles;

uint32_t start_page_size;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    vc = get_arg_val<uint32_t>(rt_args_idx++);
    noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    receiver_index = get_arg_val<uint32_t>(rt_args_idx++);

    page_size = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    num_blocks = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    block_num_tiles = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));

    start_page_size = page_size[0];

    experimental::RemoteCircularBuffer remote_cb{remote_cb_id};
    experimental::Noc noc;

    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t curr_page_size = page_size[l];
        uint32_t curr_num_blocks = num_blocks[l];
        uint32_t curr_block_num_tiles = block_num_tiles[l];

        uint32_t curr_block_size = curr_block_num_tiles * curr_page_size;
        remote_cb.set_sender_page_size(noc, curr_block_size);

        for (uint32_t block = 0; block < curr_num_blocks; ++block) {
            remote_cb.wait_front(1);
            remote_cb.pop_front(noc, 1);
        }
    }
    remote_cb.commit();
}
