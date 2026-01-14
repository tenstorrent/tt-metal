// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"
#include "experimental/circular_buffer.h"

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
    mm_init(in0_cb, in1_cb, out_cb);

    experimental::CircularBuffer cb0(in0_cb);
    experimental::CircularBuffer cb1(in1_cb);
    experimental::CircularBuffer cb_out(out_cb);

    cb_out.reserve_back(num_out_tiles);
    acquire_dst();
    cb0.wait_front(num_in0_tiles);
    cb1.wait_front(num_in1_tiles);
    matmul_tiles(in0_cb, in1_cb, in0_tile_index, in1_tile_index, out_tile_index);
    pack_tile(0, out_cb);
    cb0.pop_front(num_in0_tiles);
    cb1.pop_front(num_in1_tiles);
    release_dst();
    cb_out.push_back(num_out_tiles);
}
}  // namespace NAMESPACE
