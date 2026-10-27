// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-core decode boundary, helper compute (TRISC): kernels/boundary_compute.cpp's multi-row path on this helper's
// PH pages, with the RMSNorm statistic shared across the helpers:
//   1. residual' = (slot 0 + slot 1 + ...) + residual (slot order, as the one-core boundary) -> cb_res_out;
//   2. y = residual' * gamma (FP32 -> cb_y); q = sum over this helper's pages of residual'^2; v = REDUCE_ROW(q) -> cb_v
//      (the writer sends it to every helper);
//   3. the helpers' v tiles summed in helper order (identical on every helper and device) -> s = P v; rs =
//      rsqrt(s / hidden + eps); x = y * rs (column broadcast) -> cb_x.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
// operand / output formats for the next op (the full hardware startup runs once, at the start)
inline void cfg(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_res_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = get_compile_time_arg_val(7);
    constexpr uint32_t PH = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t inv_n = get_compile_time_arg_val(10);
    constexpr uint32_t eps = get_compile_time_arg_val(11);
    constexpr uint32_t cb_y = get_compile_time_arg_val(12);
    constexpr uint32_t cb_p = get_compile_time_arg_val(13);
    constexpr uint32_t cb_rsum = get_compile_time_arg_val(14);
    constexpr uint32_t cb_v = get_compile_time_arg_val(15);
    constexpr uint32_t cb_xchg = get_compile_time_arg_val(16);
    constexpr uint32_t helpers = get_compile_time_arg_val(17);
    constexpr uint32_t step = 4;

    // 1. residual'
    compute_kernel_hw_startup(slots > 0 ? cb_recv : cb_res, cb_res, cb_res_out);
    cb_wait_front(cb_res, PH);
    if constexpr (slots > 0) {
        cb_wait_front(cb_recv, slots * PH);
    }
    cb_reserve_back(cb_res_out, PH);
    for (uint32_t t0 = 0; t0 < PH; t0 += step) {
        const uint32_t n = PH - t0 < step ? PH - t0 : step;
        tile_regs_acquire();
        if constexpr (slots > 1) {
            add_init(cb_recv, cb_recv);
            for (uint32_t i = 0; i < n; ++i) {
                add_tiles(cb_recv, cb_recv, t0 + i, PH + t0 + i, i);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv);
            for (uint32_t s = 2; s < slots; ++s) {
                for (uint32_t i = 0; i < n; ++i) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv, s * PH + t0 + i, i);
                }
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res);
            for (uint32_t i = 0; i < n; ++i) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res, t0 + i, i);
            }
        } else if constexpr (slots == 1) {
            add_init(cb_recv, cb_res);
            for (uint32_t i = 0; i < n; ++i) {
                add_tiles(cb_recv, cb_res, t0 + i, t0 + i, i);
            }
        } else {
            copy_init(cb_res);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_res, t0 + i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_res_out);
        }
        tile_regs_release();
    }
    cb_push_back(cb_res_out, PH);
    cb_pop_front(cb_res, PH);
    if constexpr (slots > 0) {
        cb_pop_front(cb_recv, slots * PH);
    }

    // 2. q, v (the writer sends v to every helper), then y while the other helpers' v arrive
    cb_wait_front(cb_res_out, PH);
    tile_regs_acquire();
    mul_init(cb_res_out, cb_res_out);
    for (uint32_t p = 0; p < PH; ++p) {
        mul_tiles(cb_res_out, cb_res_out, p, p, 0);  // ELWMUL accumulates into DST 0
    }
    tile_regs_commit();
    cb_reserve_back(cb_sq, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_sq);
    pack_tile(0, cb_sq);
    tile_regs_release();
    cb_push_back(cb_sq, 1);

    cb_wait_front(cb_sq, 1);
    cb_wait_front(cb_scaler, 1);
    cfg(cb_scaler, cb_sq, cb_v);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_v);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, 0, 0, 0);
    reduce_uninit(cb_sq);
    tile_regs_commit();
    cb_reserve_back(cb_v, 1);
    tile_regs_wait();
    pack_tile(0, cb_v);
    tile_regs_release();
    cb_push_back(cb_v, 1);
    cb_pop_front(cb_sq, 1);
    cb_pop_front(cb_scaler, 1);
    cfg(cb_res_out, cb_gamma, cb_y);
    cb_wait_front(cb_gamma, PH);
    cb_reserve_back(cb_y, PH);
    for (uint32_t p0 = 0; p0 < PH; p0 += step) {
        const uint32_t n = PH - p0 < step ? PH - p0 : step;
        tile_regs_acquire();
        mul_init(cb_res_out, cb_gamma);
        for (uint32_t t = 0; t < n; ++t) {
            mul_tiles(cb_res_out, cb_gamma, p0 + t, p0 + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_y);
        for (uint32_t t = 0; t < n; ++t) {
            pack_tile(t, cb_y);
        }
        tile_regs_release();
    }
    cb_push_back(cb_y, PH);
    cb_pop_front(cb_gamma, PH);

    // 3. sum of the helpers' v (helper order) -> s = P v -> rs -> x
    {
        cb_wait_front(cb_xchg, helpers);
    }
    cfg(cb_xchg, cb_xchg, cb_rsum);
    tile_regs_acquire();
    if constexpr (helpers == 1) {
        copy_init(cb_xchg);
        copy_tile(cb_xchg, 0, 0);
    } else {
        add_init(cb_xchg, cb_xchg);
        add_tiles(cb_xchg, cb_xchg, 0, 1, 0);
        if constexpr (helpers > 2) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_xchg);
            for (uint32_t h = 2; h < helpers; ++h) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_xchg, h, 0);
            }
        }
    }
    tile_regs_commit();
    cb_reserve_back(cb_rsum, 1);
    tile_regs_wait();
    pack_tile(0, cb_rsum);
    tile_regs_release();
    cb_push_back(cb_rsum, 1);
    cb_pop_front(cb_xchg, helpers);

    cb_wait_front(cb_p, 1);
    cb_wait_front(cb_rsum, 1);
    cfg(cb_rsum, cb_p, cb_rs);
    tile_regs_acquire();
    matmul_init(cb_p, cb_rsum);
    matmul_tiles(cb_p, cb_rsum, 0, 0, 0);
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        0,
        VectorMode::RC,
        inv_n));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        0,
        VectorMode::RC,
        eps));
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false), 0, VectorMode::RC));
    tile_regs_commit();
    cb_reserve_back(cb_rs, 1);
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_rsum, 1);
    cb_pop_front(cb_p, 1);

    cb_wait_front(cb_rs, 1);
    cb_wait_front(cb_y, PH);
    cb_reserve_back(cb_x, PH);
    cfg(cb_y, cb_rs, cb_x);
    for (uint32_t p0 = 0; p0 < PH; p0 += step) {
        const uint32_t n = PH - p0 < step ? PH - p0 : step;
        tile_regs_acquire();
        mul_bcast_cols_init(cb_y, cb_rs);
        for (uint32_t t = 0; t < n; ++t) {
            mul_tiles_bcast_cols(cb_y, cb_rs, p0 + t, 0, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_x);
        for (uint32_t t = 0; t < n; ++t) {
            pack_tile(t, cb_x);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x, PH);
    cb_pop_front(cb_y, PH);
    cb_pop_front(cb_rs, 1);
}
