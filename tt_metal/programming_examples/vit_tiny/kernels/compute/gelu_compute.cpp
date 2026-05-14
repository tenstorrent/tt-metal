// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);
    gelu_tile_init<true>();

    for (uint32_t i = 0; i < n_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        copy_tile(cb_in, 0, 0);
        gelu_tile<true>(0);
        cb_pop_front(cb_in, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
