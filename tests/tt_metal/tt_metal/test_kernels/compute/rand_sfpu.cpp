// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t seed = get_compile_time_arg_val(1);
    constexpr uint32_t from = get_compile_time_arg_val(2);
    constexpr uint32_t scale = get_compile_time_arg_val(3);
    constexpr uint32_t dst_index = 0;
    constexpr uint32_t onetile = 1;
    // Outside every tested range, so a Dest element rand_tile does not write fails the range check.
    constexpr float sentinel = 100.0f;

    CircularBuffer cb_out(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_16, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_16);
    fill_tile_init();
    rand_tile_init(seed);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        cb_out.reserve_back(onetile);
        tile_regs_acquire();
        fill_tile(dst_index, sentinel);
        rand_tile(dst_index, from, scale);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_index, tt::CBIndex::c_16);
        tile_regs_release();
        cb_out.push_back(onetile);
    }
}
