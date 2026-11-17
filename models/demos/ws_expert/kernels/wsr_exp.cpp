// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head, compute: e = exp(s) for each row tile's picks tile (CB 1, fp32: the picks' logits minus the token's
// largest, -inf past the picks) -> CB 2, fp32 throughout (unpacked to DEST in fp32). RT: 0 row tiles.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t s_cb = tt::CBIndex::c_1, e_cb = tt::CBIndex::c_2;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(s_cb, e_cb);
    copy_init(s_cb);
    exp_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(s_cb, 1);
        cb_reserve_back(e_cb, 1);
        tile_regs_acquire();
        copy_tile(s_cb, 0, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, e_cb);
        tile_regs_release();
        cb_pop_front(s_cb, 1);
        cb_push_back(e_cb, 1);
    }
}
