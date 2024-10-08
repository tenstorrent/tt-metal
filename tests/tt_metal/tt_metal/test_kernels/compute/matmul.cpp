// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {

    uint32_t block_tile_dim = get_compile_time_arg_val(0);
    uint32_t dst_tile_rows = get_compile_time_arg_val(1);
    uint32_t dst_tile_cols = get_compile_time_arg_val(2);
    uint32_t block_cnt = get_compile_time_arg_val(3);
    uint32_t in0_block_tile_cnt = get_compile_time_arg_val(4);
    uint32_t in1_block_tile_cnt = get_compile_time_arg_val(5);
    uint32_t out_block_tile_cnt = get_compile_time_arg_val(6);

    mm_init();

    acquire_dst();
    for(uint32_t b=0;b<block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, in0_block_tile_cnt);
        cb_wait_front(tt::CB::c_in1, in1_block_tile_cnt);
        int dst_tile_index = 0;
        int in0_block_tile_index = 0;
        for(uint32_t r=0;r<dst_tile_rows;++r)
        {
            for(uint32_t c=0;c<dst_tile_cols;++c)
            {
                int in1_block_tile_index = 0;
                for(uint32_t i=0;i<block_tile_dim;++i)
                {
                    matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, in0_block_tile_index+i, in1_block_tile_index+c, dst_tile_index, false);
                    in1_block_tile_index += dst_tile_cols;
                }
                dst_tile_index++;
            }
            in0_block_tile_index += block_tile_dim;
        }
        cb_pop_front(tt::CB::c_in0, in0_block_tile_cnt);
        cb_pop_front(tt::CB::c_in1, in1_block_tile_cnt);
    }

    // Pack out
    cb_reserve_back(tt::CB::c_out0, out_block_tile_cnt);
    for(uint32_t i=0 ; i<out_block_tile_cnt;++i)
    {
        pack_tile(i, tt::CB::c_out0);
    }

    cb_push_back(tt::CB::c_out0, out_block_tile_cnt);

    release_dst();
}
}
