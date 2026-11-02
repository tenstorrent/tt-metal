// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill rotation, compute. H_1024 = H32 (x) H32, so output tile i of block b is
//   y_i = sum_j x_j @ (h(j,i) diag(d_bj) H32 / 32),  h(j,i) = (-1)^popcount(i & j),
// 32 tile matmuls per output tile, with all 64 +-tiles of the block resident in cb_pn.
// Runtime args: u0, nu.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_pn = 1, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_pn, cb_out);  // matmul: in0 -> SrcB, in1 -> SrcA
    matmul_init(cb_x, cb_pn);
    uint32_t cur_b = 0xffffffff;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t b = u / R;
        if (b != cur_b) {
            if (cur_b != 0xffffffff) {
                cb_pop_front(cb_pn, 64);
            }
            cb_wait_front(cb_pn, 64);
            cur_b = b;
        }
        cb_wait_front(cb_x, 32);
        for (uint32_t i = 0; i < 32; ++i) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < 32; ++j) {
                matmul_tiles(cb_x, cb_pn, j, 2 * j + (__builtin_popcount(i & j) & 1), 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_x, 32);
    }
    if (cur_b != 0xffffffff) {
        cb_pop_front(cb_pn, 64);
    }
}
