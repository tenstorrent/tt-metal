// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Residual add + RMSNorm, compute: s = x + b over this core's W tiles of a row tile (W even); its part of
// the row sums of s^2 (cb_part); then, from the Q parts of the row's cores (cb_recv),
//   h = s * rsqrt(sum / H + eps)
// Compile-time args: 0 W, 1 Q, 2 eps (fp32 bits), 3 1 / H (fp32 bits).

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
constexpr uint32_t cb_x = 0, cb_b = 1, cb_scaler = 2, cb_part = 3, cb_recv = 4, cb_rs = 5, cb_sq = 6, cb_s = 16,
                   cb_h = 17;

inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t eps = get_compile_time_arg_val(2);
    constexpr uint32_t inv_h = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_x, cb_b, cb_s);
    // s = x + b and s^2, two tiles per DEST acquire (the add runs twice, the second copy is squared)
    constexpr uint32_t step = 2;
    static_assert(W % step == 0, "tiles in pairs");
    add_tiles_init(cb_x, cb_b);
    square_tile_init();
    for (uint32_t i = 0; i < W; i += step) {
        cb_wait_front(cb_x, step);
        cb_wait_front(cb_b, step);
        cb_reserve_back(cb_s, step);
        cb_reserve_back(cb_sq, step);
        tile_regs_acquire();
        for (uint32_t t = 0; t < step; t++) {
            add_tiles(cb_x, cb_b, t, t, t);
            add_tiles(cb_x, cb_b, t, t, step + t);
            square_tile(step + t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < step; t++) {
            pack_tile(t, cb_s);
            pack_tile(step + t, cb_sq);
        }
        tile_regs_release();
        cb_push_back(cb_s, step);
        cb_push_back(cb_sq, step);
        cb_pop_front(cb_x, step);
        cb_pop_front(cb_b, step);
    }
    cb_wait_front(cb_sq, W);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_part, 1);
    reconfig_data_format(cb_scaler, cb_sq);
    pack_reconfig_data_format(cb_part);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_part);
    tile_regs_acquire();
    for (uint32_t i = 0; i < W; i++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, i, 0, 0);
    }
    reduce_uninit();
    pack_one(cb_part);
    cb_push_back(cb_part, 1);
    cb_pop_front(cb_sq, W);

    // rs = rsqrt(sum of the row's parts / H + eps)
    cb_wait_front(cb_recv, Q);
    cb_reserve_back(cb_rs, 1);
    reconfig_data_format_srca(cb_recv);
    pack_reconfig_data_format(cb_rs);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_recv);
    copy_tile(cb_recv, 0, 0);
    for (uint32_t q = 1; q < Q; q++) {
        copy_tile_to_dst_init_short(cb_recv);
        copy_tile(cb_recv, q, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_one(cb_rs);
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_recv, Q);

    // h = s * rs; the writer stores s and h, then frees both
    cb_wait_front(cb_rs, 1);
    cb_wait_front(cb_s, W);
    reconfig_data_format(cb_s, cb_rs);
    pack_reconfig_data_format(cb_h);
    mul_bcast_cols_init(cb_s, cb_rs);
    for (uint32_t i = 0; i < W; i += step) {
        cb_reserve_back(cb_h, step);
        tile_regs_acquire();
        for (uint32_t t = 0; t < step; t++) {
            mul_tiles_bcast_cols(cb_s, cb_rs, i + t, 0, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < step; t++) {
            pack_tile(t, cb_h);
        }
        tile_regs_release();
        cb_push_back(cb_h, step);
    }
    cb_pop_front(cb_rs, 1);
}
