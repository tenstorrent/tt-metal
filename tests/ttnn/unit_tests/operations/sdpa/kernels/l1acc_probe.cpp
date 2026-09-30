// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf research (not for merge): sum N input tiles into one output tile with packer L1 accumulate.
// The dest mode comes from the compute config; the output CB format decides the L1 accumulation format.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_tile_to_dst_init_short(cb_in);
    cb_reserve_back(cb_out, 1);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (i == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        pack_tile<true>(0, cb_out, 0);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    cb_push_back(cb_out, 1);
}
