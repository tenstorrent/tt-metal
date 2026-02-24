// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        tile_regs_release();
    }
}
