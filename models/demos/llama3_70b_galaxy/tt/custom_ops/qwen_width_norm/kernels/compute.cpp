// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// qwen_width_norm compute, per tile row of this core's width chunk (w tiles, w % DB == 0):
//   1. CB_XSQ <- x * x (DB tiles per DST pass), then CB_PART <- reduce_row_sum(CB_XSQ) * 1/W: the stock RMSNorm
//      sequence of qwen_q_prep norm_compute.cpp (square_row, mean_sq_row) over w tiles.
//   2. wait for the gx partials of the row group (CB_GATH, pushed by the reader) and add them in slot order
//      (CB_ACC, one add_tiles per partial), then rsqrt(sum + eps) -> CB_RSTD (norm_compute.cpp rstd_row).
//   3. y = (x * bcast_col(rstd)) -> fp32 CB_FUSION -> * bcast_row(gamma) -> bf8 CB_OUT, DB tiles at a time
//      (norm_compute.cpp normalize_row).
// fp32 DEST (DB = 4 tiles per half-sync pass), HiFi2 as the stock norm.
//
// Compile-time args: wnorm_common.hpp prefix (no per-core args: the work comes from the logical coords)

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "wnorm_common.hpp"

using namespace wnorm;

// A row occupies maxw slots of CB_X and CB_XSQ (w <= maxw used), so every row starts at a slot multiple of maxw
// and indexed reads never wrap the CB end.
template <uint32_t maxw>
FORCE_INLINE void square_row(uint32_t w) {
    reconfig_data_format(CB_X, CB_X);
    pack_reconfig_data_format(CB_XSQ);
    mul_init(CB_X, CB_X, false);
    cb_wait_front(CB_X, maxw);
    cb_reserve_back(CB_XSQ, maxw);
    for (uint32_t t = 0; t < w; t += DB) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < DB; ++i) {
            mul_tiles(CB_X, CB_X, t + i, t + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < DB; ++i) {
            pack_tile<true>(i, CB_XSQ, t + i);
        }
        tile_regs_release();
    }
    cb_push_back(CB_XSQ, maxw);
}

template <uint32_t inv_w, uint32_t maxw>
FORCE_INLINE void partial_row(uint32_t w) {
    reconfig_data_format(CB_X, CB_XSQ, CB_X, CB_SCALER);
    cb_wait_front(CB_SCALER, 1);
    reconfig_data_format(CB_XSQ, CB_SCALER);
    tile_regs_acquire();
    reconfig_data_format(CB_SCALER, CB_XSQ);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XSQ, CB_SCALER, CB_PART);
    cb_wait_front(CB_XSQ, maxw);
    for (uint32_t j = 0; j < w; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XSQ, CB_SCALER, j, 0, 0);
    }
    cb_pop_front(CB_XSQ, maxw);
    reduce_uninit();
    reconfig_data_format(CB_XSQ, CB_SCALER);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_w);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_PART, 1);
    pack_reconfig_data_format(CB_PART);
    pack_tile(0, CB_PART);
    tile_regs_release();
    cb_push_back(CB_PART, 1);
}

// CB_ACC <- a[ia] + b[ib] (one fp32 tile).
FORCE_INLINE void add_into_acc(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(CB_ACC);
    add_init(a, b, false);
    tile_regs_acquire();
    add_tiles(a, b, ia, ib, 0);
    tile_regs_commit();
    if (a == CB_ACC) {
        cb_pop_front(CB_ACC, 1);
    }
    cb_reserve_back(CB_ACC, 1);
    tile_regs_wait();
    pack_tile(0, CB_ACC);
    tile_regs_release();
    cb_push_back(CB_ACC, 1);
    cb_wait_front(CB_ACC, 1);
}

template <uint32_t gx>
FORCE_INLINE void combine_row() {
    cb_wait_front(CB_GATH, gx);
    add_into_acc(CB_GATH, 0, CB_GATH, 1);
    for (uint32_t k = 2; k < gx; ++k) {
        add_into_acc(CB_ACC, 0, CB_GATH, k);
    }
    cb_pop_front(CB_GATH, gx);

    // rstd = rsqrt(sum + eps)
    reconfig_data_format(CB_ACC, CB_EPS);
    pack_reconfig_data_format(CB_RSTD);
    tile_regs_acquire();
    add_init(CB_ACC, CB_EPS, false);
    add_tiles(CB_ACC, CB_EPS, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    cb_reserve_back(CB_RSTD, 1);
    tile_regs_wait();
    pack_tile(0, CB_RSTD);
    tile_regs_release();
    cb_push_back(CB_RSTD, 1);
    cb_pop_front(CB_ACC, 1);
}

template <uint32_t maxw>
FORCE_INLINE void normalize_row(uint32_t w) {
    cb_wait_front(CB_RSTD, 1);
    cb_wait_front(CB_GAMMA, w);
    for (uint32_t t = 0; t < w; t += DB) {
        reconfig_data_format(CB_X, CB_RSTD);
        pack_reconfig_data_format(CB_FUSION);
        mul_bcast_cols_init(CB_X, CB_RSTD);
        tile_regs_acquire();
        for (uint32_t i = 0; i < DB; ++i) {
            mul_tiles_bcast_cols(CB_X, CB_RSTD, t + i, 0, i);
        }
        tile_regs_commit();
        cb_reserve_back(CB_FUSION, DB);
        tile_regs_wait();
        for (uint32_t i = 0; i < DB; ++i) {
            pack_tile(i, CB_FUSION);
        }
        tile_regs_release();
        cb_push_back(CB_FUSION, DB);

        reconfig_data_format_srca(CB_X, CB_FUSION);
        pack_reconfig_data_format(CB_OUT);
        reconfig_data_format_srcb(CB_RSTD, CB_GAMMA);
        mul_bcast_rows_init(CB_FUSION, CB_GAMMA);
        cb_wait_front(CB_FUSION, DB);
        tile_regs_acquire();
        for (uint32_t i = 0; i < DB; ++i) {
            mul_tiles_bcast_rows(CB_FUSION, CB_GAMMA, i, t + i, i);
        }
        tile_regs_commit();
        cb_reserve_back(CB_OUT, DB);
        tile_regs_wait();
        for (uint32_t i = 0; i < DB; ++i) {
            pack_tile(i, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, DB);
        cb_pop_front(CB_FUSION, DB);
    }
    cb_pop_front(CB_RSTD, 1);
    cb_pop_front(CB_X, maxw);
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t gx = get_compile_time_arg_val(1);
    constexpr uint32_t gy = get_compile_time_arg_val(2);
    constexpr uint32_t Rt = get_compile_time_arg_val(3);
    constexpr uint32_t maxw = get_compile_time_arg_val(4);
    constexpr uint32_t inv_w = get_compile_time_arg_val(6);
    const Work wk = core_work(get_absolute_logical_x(), get_absolute_logical_y(), Wt, gx, Rt, gy);
    const uint32_t num_rows = wk.n;
    const uint32_t w = wk.w;

    compute_kernel_hw_startup(CB_X, CB_X, CB_XSQ);
    cb_wait_front(CB_EPS, 1);

    for (uint32_t r = 0; r < num_rows; ++r) {
        square_row<maxw>(w);
        partial_row<inv_w, maxw>(w);
        combine_row<gx>();
        normalize_row<maxw>(w);
    }

    // The reader pushes the scaler, eps and gamma once (also when this core has no rows).
    cb_wait_front(CB_SCALER, 1);
    cb_wait_front(CB_GAMMA, w);
    cb_pop_front(CB_SCALER, 1);
    cb_pop_front(CB_EPS, 1);
    cb_pop_front(CB_GAMMA, w);
}
