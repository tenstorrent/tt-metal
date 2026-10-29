// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// q / k head norm + partial RoPE, compute, per group (one head's 256 dims of 32 rows, 8 tiles):
//   y = t * rsqrt(mean(t^2) + eps) * w
//   rotated dims 0..63 (tiles 0, 1 = halves a, b): a' = a cos - b sin, b' = b cos + a sin
// Compile-time args: 0 groups, 1 eps (fp32 bits).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

namespace {
constexpr uint32_t cb_t = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4, cb_sq = 5, cb_rs = 6, cb_n = 7, cb_y = 8,
                   cb_out = 16, Dt = 8;

inline void pack_at(uint32_t cb, uint32_t idx) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb, idx);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t groups = get_compile_time_arg_val(0);
    constexpr uint32_t eps = get_compile_time_arg_val(1);
    constexpr uint32_t inv_d = 0x3b800000;  // 1 / 256

    compute_kernel_hw_startup(cb_t, cb_t, cb_sq);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_w, Dt);
    for (uint32_t g = 0; g < groups; g++) {
        cb_wait_front(cb_t, Dt);
        // rs = rsqrt(mean of t^2 over the 256 dims + eps)
        cb_reserve_back(cb_sq, Dt);
        reconfig_data_format(cb_t, cb_t);
        pack_reconfig_data_format(cb_sq);
        mul_tiles_init(cb_t, cb_t);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles(cb_t, cb_t, d, d, 0);
            pack_at(cb_sq, d);
        }
        cb_push_back(cb_sq, Dt);
        cb_wait_front(cb_sq, Dt);
        cb_reserve_back(cb_rs, 1);
        reconfig_data_format(cb_scaler, cb_sq);
        pack_reconfig_data_format(cb_rs);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_rs);
        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; d++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, d, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_at(cb_rs, 0);
        cb_push_back(cb_rs, 1);
        cb_pop_front(cb_sq, Dt);

        // n = t * rs
        cb_wait_front(cb_rs, 1);
        cb_reserve_back(cb_n, Dt);
        reconfig_data_format(cb_t, cb_rs);
        pack_reconfig_data_format(cb_n);
        mul_bcast_cols_init(cb_t, cb_rs);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_t, cb_rs, d, 0, 0);
            pack_at(cb_n, d);
        }
        cb_push_back(cb_n, Dt);
        cb_pop_front(cb_rs, 1);
        cb_pop_front(cb_t, Dt);

        // y = n * w: the rotated tiles to cb_y, the rest straight to their output slots
        cb_wait_front(cb_n, Dt);
        cb_reserve_back(cb_y, 2);
        cb_reserve_back(cb_out, Dt);
        reconfig_data_format(cb_n, cb_w);
        mul_bcast_rows_init(cb_n, cb_w);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_n, cb_w, d, d, 0);
            if (d < 2) {
                pack_reconfig_data_format(cb_y);
                pack_at(cb_y, d);
            } else {
                pack_reconfig_data_format(cb_out);
                pack_at(cb_out, d);
            }
        }
        cb_push_back(cb_y, 2);
        cb_pop_front(cb_n, Dt);

        // a' = a cos - b sin, b' = b cos + a sin
        cb_wait_front(cb_y, 2);
        cb_wait_front(cb_cos, 1);
        cb_wait_front(cb_sin, 1);
        pack_reconfig_data_format(cb_out);
        for (uint32_t half = 0; half < 2; half++) {
            const uint32_t self = half, other = 1 - half;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_y);
            copy_tile_to_dst_init_short(cb_y);
            copy_tile(cb_y, self, 0);
            reconfig_data_format_srca(cb_cos);
            copy_tile_to_dst_init_short(cb_cos);
            copy_tile(cb_cos, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            reconfig_data_format_srca(cb_y);
            copy_tile_to_dst_init_short(cb_y);
            copy_tile(cb_y, other, 2);
            reconfig_data_format_srca(cb_sin);
            copy_tile_to_dst_init_short(cb_sin);
            copy_tile(cb_sin, 0, 3);
            mul_binary_tile_init();
            mul_binary_tile(2, 3, 2);
            if (half == 0) {
                sub_binary_tile_init();
                sub_binary_tile(0, 2, 0);
            } else {
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
            }
            pack_at(cb_out, half);
        }
        cb_push_back(cb_out, Dt);
        cb_pop_front(cb_y, 2);
        cb_pop_front(cb_cos, 1);
        cb_pop_front(cb_sin, 1);
    }
}
