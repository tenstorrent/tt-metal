// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN gates, compute: per row tile and 32-head column tile c
//   g_c = neg_a_c * softplus(a_c + dt_c) * mask,  beta_c = sigmoid(b_c) * mask
// (dt, neg_a broadcast down the rows, mask elementwise). Compile-time args: 0 row tiles of this core.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace {
constexpr uint32_t cb_a = 1, cb_b = 2, cb_dt = 3, cb_na = 4, cb_m = 5, cb_t1 = 6, cb_t2 = 7, cb_g = 16, cb_beta = 17;

inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}

// out = t * mask
inline void masked(uint32_t t, uint32_t out) {
    cb_wait_front(t, 1);
    cb_reserve_back(out, 1);
    reconfig_data_format(t, cb_m);
    pack_reconfig_data_format(out);
    mul_tiles_init(t, cb_m);
    tile_regs_acquire();
    mul_tiles(t, cb_m, 0, 0, 0);
    pack_one(out);
    cb_pop_front(t, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_a, cb_dt, cb_t1);
    cb_wait_front(cb_dt, 2);
    cb_wait_front(cb_na, 2);
    for (uint32_t r = 0; r < rows; r++) {
        cb_wait_front(cb_m, 1);
        cb_wait_front(cb_a, 2);
        cb_wait_front(cb_b, 2);
        for (uint32_t c = 0; c < 2; c++) {
            // softplus(a + dt)
            cb_reserve_back(cb_t1, 1);
            reconfig_data_format(cb_a, cb_dt);
            pack_reconfig_data_format(cb_t1);
            add_bcast_rows_init(cb_a, cb_dt);
            tile_regs_acquire();
            add_tiles_bcast_rows(cb_a, cb_dt, c, c, 0);
            softplus_tile_init();
            softplus_tile(0, 0x3f800000, 0x3f800000, 0x41a00000);  // beta 1, 1 / beta, threshold 20
            pack_one(cb_t1);
            // * neg_a
            cb_wait_front(cb_t1, 1);
            cb_reserve_back(cb_t2, 1);
            reconfig_data_format(cb_t1, cb_na);
            pack_reconfig_data_format(cb_t2);
            mul_bcast_rows_init(cb_t1, cb_na);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_t1, cb_na, 0, c, 0);
            pack_one(cb_t2);
            cb_pop_front(cb_t1, 1);
            masked(cb_t2, cb_g);
        }
        for (uint32_t c = 0; c < 2; c++) {
            cb_reserve_back(cb_t1, 1);
            reconfig_data_format_srca(cb_b);
            pack_reconfig_data_format(cb_t1);
            copy_tile_to_dst_init_short(cb_b);
            tile_regs_acquire();
            copy_tile(cb_b, c, 0);
            sigmoid_tile_init();
            sigmoid_tile(0);
            pack_one(cb_t1);
            masked(cb_t1, cb_beta);
        }
        cb_pop_front(cb_a, 2);
        cb_pop_front(cb_b, 2);
        cb_pop_front(cb_m, 1);
    }
}
