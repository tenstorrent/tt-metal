// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t seed = get_compile_time_arg_val(1);
    constexpr uint32_t from = get_compile_time_arg_val(2);
    constexpr uint32_t scale = get_compile_time_arg_val(3);

    CircularBuffer cb_out(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_16, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_16);
    rand_tile_init(seed);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        cb_out.reserve_back(1);
        tile_regs_acquire();
        rand_tile(0, from, scale);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();
        cb_out.push_back(1);
    }
}
