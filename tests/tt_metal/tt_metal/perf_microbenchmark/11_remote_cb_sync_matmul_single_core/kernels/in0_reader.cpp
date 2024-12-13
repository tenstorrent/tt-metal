// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

constexpr uint32_t num_layers = get_compile_time_arg_val(0);

void kernel_main() {
    uint32_t rt_args_idx = 0;
    tt_l1_ptr uint32_t* num_blocks = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    tt_l1_ptr uint32_t* in0_block_tiles =
        (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));
    tt_l1_ptr uint32_t* out_block_num_tiles =
        (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_layers)));

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t out_cb_id = 16;

    for (uint32_t l = 0; l < num_layers; ++l) {
        uint32_t curr_num_blocks = num_blocks[l];
        uint32_t curr_block_num_tiles = in0_block_tiles[l];
        uint32_t curr_out_block_num_tiles = out_block_num_tiles[l];

        for (uint32_t block = 0; block < curr_num_blocks; block++) {
            cb_reserve_back(cb_id_in0, curr_block_num_tiles);
            cb_push_back(cb_id_in0, curr_block_num_tiles);
        }

        cb_wait_front(out_cb_id, curr_out_block_num_tiles);
        cb_pop_front(out_cb_id, curr_out_block_num_tiles);
    }
}
