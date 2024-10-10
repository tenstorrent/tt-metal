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
    const uint32_t num_in0_tiles = 1;
    const uint32_t num_in1_tiles = 1;
    const uint32_t num_out_tiles = 1;
    const uint32_t in0_tile_index = 0;
    const uint32_t in1_tile_index = 0;
    const uint32_t out_tile_index = 0;
    const bool transpose = false;
    mm_init();
    cb_reserve_back(out_cb, num_out_tiles);
    acquire_dst();
    cb_wait_front(in0_cb, num_in0_tiles);
    cb_wait_front(in1_cb, num_in1_tiles);
    matmul_tiles(in0_cb, in1_cb, in0_tile_index, in1_tile_index, out_tile_index, transpose);
    pack_tile(0, out_cb);
    cb_pop_front(in0_cb, num_in0_tiles);
    cb_pop_front(in1_cb, num_in1_tiles);
    release_dst();
    cb_push_back(out_cb, num_out_tiles);
}
}  // namespace NAMESPACE
