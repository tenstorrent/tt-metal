// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 4;
    compute_kernel_hw_startup(cb_in, cb_out);
    const uint32_t blocks = get_arg_val<uint32_t>(0);
    tile_regs_acquire();
    // CB0 is explicitly UnpackToDestFp32: copy bypasses the reduced-mantissa
    // SrcA route. SFPU preserves the original partial + accumulated order.
    for (uint32_t block = 0; block < blocks; ++block) {
        cb_wait_front(cb_in, 1);
        copy_init(cb_in);
        copy_tile(cb_in, 0, block == 0 ? 0 : 1);
        if (block != 0) {
            add_binary_tile_init();
            add_binary_tile(1, 0, 0);
        }
        cb_pop_front(cb_in, 1);
    }
    cb_reserve_back(cb_out, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
