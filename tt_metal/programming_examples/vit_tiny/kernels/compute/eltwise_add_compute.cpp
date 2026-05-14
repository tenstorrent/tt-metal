// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < n_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
