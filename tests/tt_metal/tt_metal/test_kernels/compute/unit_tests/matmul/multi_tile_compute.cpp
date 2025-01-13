// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    const uint32_t out_cb = get_compile_time_arg_val(2);
    const uint32_t in0_num_tiles = get_compile_time_arg_val(3);
    const uint32_t in1_num_tiles = get_compile_time_arg_val(4);
    const uint32_t out_num_tiles = get_compile_time_arg_val(5);
    const uint32_t out_r = get_compile_time_arg_val(6);
    const uint32_t out_c = get_compile_time_arg_val(7);
    const uint32_t in0_k = get_compile_time_arg_val(8);
    const bool transpose = false;

    // we are looking at block
    // out = in0[r x k]*in1[k x c]
    mm_init();
    acquire_dst();

    uint32_t out_tile_index = 0;
    uint32_t in0_index_r_offset = 0;
    cb_wait_front(in0_cb, in0_num_tiles);
    cb_wait_front(in1_cb, in1_num_tiles);
    for (uint32_t r = 0; r < out_r; r++) {
        for (uint32_t c = 0; c < out_c; c++) {
            uint32_t in1_index_c_offset = 0;
            for (uint32_t k = 0; k < in0_k; k++) {
                int in0_tile_index = in0_index_r_offset + k;
                int in1_tile_index = in1_index_c_offset + c;
                matmul_tiles(in0_cb, in1_cb, in0_tile_index, in1_tile_index, out_tile_index, transpose);
                in1_index_c_offset += k;
            }
            out_tile_index++;
        }
        in0_index_r_offset += in0_k;
    }
    cb_pop_front(in0_cb, in0_num_tiles);
    cb_pop_front(in1_cb, in1_num_tiles);

    cb_reserve_back(out_cb, out_num_tiles);
    for (uint32_t tile_index = 0; tile_index < out_num_tiles; tile_index++) {
        pack_tile(tile_index, out_cb);
    }
    cb_push_back(out_cb, out_num_tiles);
    release_dst();
}
}  // namespace NAMESPACE
