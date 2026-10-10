// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE down + expert-sum compute (Laguna): DST 0 accumulates x_u @ Wd_u[:, nt] over every active
// expert u (the routing weights are already applied to x_u), one packed bf16 tile. No active expert: no output
// (the writer stores a zero tile).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt_sh = get_compile_time_arg_val(1);  // shared expert unit (0: none), weights in cb_w_sh
    constexpr uint32_t cb_w_sh = 6;
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_meta = 2;
    constexpr uint32_t cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (n == 0 && Kt_sh == 0) {
        return;
    }
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t u = 0; u < n; ++u) {
        cb_wait_front(cb_x, Kt);
        cb_wait_front(cb_w, Kt);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            matmul_tiles(cb_x, cb_w, kt, kt, 0);
        }
        cb_pop_front(cb_x, Kt);
        cb_pop_front(cb_w, Kt);
    }
    if constexpr (Kt_sh > 0) {
        reconfig_data_format(cb_w_sh, cb_x);
        matmul_init(cb_x, cb_w_sh);
        cb_wait_front(cb_x, Kt_sh);
        cb_wait_front(cb_w_sh, Kt_sh);
        for (uint32_t kt = 0; kt < Kt_sh; ++kt) {
            matmul_tiles(cb_x, cb_w_sh, kt, kt, 0);
        }
        cb_pop_front(cb_x, Kt_sh);
        cb_pop_front(cb_w_sh, Kt_sh);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
