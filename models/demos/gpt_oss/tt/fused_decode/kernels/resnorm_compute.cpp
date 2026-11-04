// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash residual add + RMSNorm, compute (resnorm_reader.cpp): per core's NB column tiles j
//   x_j = r_j + y_j (BF16, into the new residual's shard cb_r2);
//   partial = rowsum_j (x_j^2) (FP32, column 0; the row's GX partials are exchanged by the writers);
//   rs = rsqrt(sum of the GX partials / hidden + eps);
//   n_j = (x_j * rs) * g_j (column broadcast of rs, row broadcast of the weights) -> cb_n (RN_OUT_BF16, BF16),
//   cb_n8 (RN_X8, BFP8), cb_nm (RN_ROUTER: BF16, this kernel's own router input);
//   RN_ROUTER: partial logits = sum_j n_j W_j (W_j: the router weights' K tile nb NB + j) -> cb_lp; on the row's core
//   0, logits = sum of the GX partials + bias -> cb_lo (BF16).
//
// compile-time args: [cb_y, cb_r, cb_g, cb_sc, cb_all, cb_r2, cb_sq, cb_part, cb_rs, cb_t, cb_n, NB, GX, inv_w bits,
// eps bits, cb_n8, cb_nm, cb_w, cb_lp, cb_lall, cb_lb, cb_lo, cb_b]
// RN_F32 (the drafter): the residual is FP32 (cb_r unpacked straight to DST, UnpackToDestFp32) and x = r + y (+ b,
// RN_BIAS: the bias tiles cb_b, every row the bias) on the SFPU in FP32.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t BLK = 4;  // tiles per DST round (FP32 DST: 4 per half)

// the sum of tiles 0 .. n - 1 of cb (FP32) into DST 0 (SFPU adds, FP32)
inline void sum_tiles(uint32_t cb, uint32_t n) {
    copy_init(cb);
    copy_tile(cb, 0, 0);
    for (uint32_t k = 1; k < n; ++k) {
        copy_init(cb);
        copy_tile(cb, k, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_all = get_compile_time_arg_val(4);
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(8);
    constexpr uint32_t cb_t = get_compile_time_arg_val(9);
    constexpr uint32_t cb_n = get_compile_time_arg_val(10);
    constexpr uint32_t NB = get_compile_time_arg_val(11);
    constexpr uint32_t GX = get_compile_time_arg_val(12);
    constexpr uint32_t inv_w = get_compile_time_arg_val(13);
    constexpr uint32_t eps = get_compile_time_arg_val(14);
    constexpr uint32_t cb_n8 = get_compile_time_arg_val(15);
    constexpr uint32_t cb_nm = get_compile_time_arg_val(16);
    constexpr uint32_t cb_w = get_compile_time_arg_val(17);
    constexpr uint32_t cb_lp = get_compile_time_arg_val(18);
    constexpr uint32_t cb_lall = get_compile_time_arg_val(19);
    constexpr uint32_t cb_lb = get_compile_time_arg_val(20);
    constexpr uint32_t cb_lo = get_compile_time_arg_val(21);
    constexpr uint32_t cb_b = get_compile_time_arg_val(22);

    compute_kernel_hw_startup(cb_r, cb_y, cb_r2);
#ifndef RN_MPC
#define RN_MPC 1
#endif
#ifdef RN_BIAS
    cb_wait_front(cb_b, NB);
#endif
    // RN_MPC row tiles per core, one after another (cb_r / cb_r2: the shards, row tile mi's NB tiles at the front)
    for (uint32_t mi = 0; mi < RN_MPC; ++mi) {
        // (the inits do not switch the unpacker formats: state_configure is a no-op outside the sentinel build)
        // x = r + y -> the new residual
        cb_wait_front(cb_r, NB);
        cb_wait_front(cb_y, NB);
        pack_reconfig_data_format(cb_r2);
#ifdef RN_F32
        for (uint32_t j = 0; j < NB; ++j) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_r);
            copy_init(cb_r);
            copy_tile(cb_r, j, 0);
            reconfig_data_format_srca(cb_y);
            copy_init(cb_y);
            copy_tile(cb_y, j, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
#ifdef RN_BIAS
        reconfig_data_format_srca(cb_b);
        copy_init(cb_b);
        copy_tile(cb_b, j, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
#endif
        tile_regs_commit();
#else
    reconfig_data_format(cb_r, cb_y);
    add_init(cb_r, cb_y);
    for (uint32_t j0 = 0; j0 < NB; j0 += BLK) {
        const uint32_t n = j0 + BLK < NB ? BLK : NB - j0;
        tile_regs_acquire();
        for (uint32_t b = 0; b < n; ++b) {
            add_tiles(cb_r, cb_y, j0 + b, j0 + b, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_r2, 1);
            pack_tile(b, cb_r2);
            cb_push_back(cb_r2, 1);
        }
        tile_regs_release();
    }
#endif
#ifdef RN_F32
        cb_reserve_back(cb_r2, 1);
        tile_regs_wait();
        pack_tile(0, cb_r2);
        tile_regs_release();
        cb_push_back(cb_r2, 1);
    }
#endif
    cb_pop_front(cb_r, NB);
    cb_pop_front(cb_y, NB);

    // x^2 (FP32)
    cb_wait_front(cb_r2, NB);
    reconfig_data_format(cb_r2, cb_r2);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_r2, cb_r2);
    for (uint32_t j0 = 0; j0 < NB; j0 += BLK) {
        const uint32_t n = j0 + BLK < NB ? BLK : NB - j0;
        tile_regs_acquire();
        for (uint32_t b = 0; b < n; ++b) {
            mul_tiles(cb_r2, cb_r2, j0 + b, j0 + b, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_sq, 1);
            pack_tile(b, cb_sq);
            cb_push_back(cb_sq, 1);
        }
        tile_regs_release();
    }

    // the partial row sums (a row-sum reduce runs as a matmul with the scaler in SrcA and the data in SrcB: formats in
    // that order)
    cb_wait_front(cb_sq, NB);
    cb_wait_front(cb_sc, 1);
    reconfig_data_format(cb_sc, cb_sq);
    pack_reconfig_data_format(cb_part);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, cb_part);
    for (uint32_t j = 0; j < NB; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, j, 0, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_part, 1);
    tile_regs_wait();
    // (packed while reduce_init's packer mask is in effect: zeros outside column 0, which the row's sum below relies
    // on; the math leaves DST's other columns as they were)
    pack_tile(0, cb_part);
    tile_regs_release();
    reduce_uninit(cb_sq);
    cb_push_back(cb_part, 1);
    cb_pop_front(cb_sq, NB);

    // rs = rsqrt(sum / hidden + eps): the GX partials (each zero outside column 0, the reduce's packer zeroing) summed
    // by one row-sum reduce over them (FP32 DST accumulation)
    cb_wait_front(cb_all, GX);
    reconfig_data_format(cb_sc, cb_all);
    pack_reconfig_data_format(cb_rs);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_all, cb_sc, cb_rs);
    for (uint32_t k = 0; k < GX; ++k) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_all, cb_sc, k, 0, 0);
    }
    reduce_uninit(cb_all);
    // (column 0 only: the SFPU over faces 0 and 2, VectorMode::C)
    binop_with_scalar_tile_init();
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        0,
        VectorMode::C,
        inv_w)));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        0,
        VectorMode::C,
        eps)));
    rsqrt_tile_init();
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false), 0, VectorMode::C)));
    tile_regs_commit();
    cb_reserve_back(cb_rs, 1);
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_all, GX);

    // t_j = x_j rs (FP32, all NB first: one init)
    cb_wait_front(cb_rs, 1);
    reconfig_data_format(cb_r2, cb_rs);
    pack_reconfig_data_format(cb_t);
    mul_bcast_cols_init(cb_r2, cb_rs);
    for (uint32_t j0 = 0; j0 < NB; j0 += BLK) {
        const uint32_t n = j0 + BLK < NB ? BLK : NB - j0;
        tile_regs_acquire();
        for (uint32_t b = 0; b < n; ++b) {
            mul_tiles_bcast_cols(cb_r2, cb_rs, j0 + b, 0, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_t, 1);
            pack_tile(b, cb_t);
            cb_push_back(cb_t, 1);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_rs, 1);
    // n_j = t_j g_j
    cb_wait_front(cb_t, NB);
    cb_wait_front(cb_g, NB);
    reconfig_data_format(cb_t, cb_g);
    mul_bcast_rows_init(cb_t, cb_g);
    for (uint32_t j0 = 0; j0 < NB; j0 += BLK) {
        const uint32_t n = j0 + BLK < NB ? BLK : NB - j0;
        tile_regs_acquire();
        for (uint32_t b = 0; b < n; ++b) {
            mul_tiles_bcast_rows(cb_t, cb_g, j0 + b, j0 + b, b);
        }
        tile_regs_commit();
        tile_regs_wait();
#ifdef RN_OUT_BF16
        pack_reconfig_data_format(cb_n);
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_n, 1);
            pack_tile(b, cb_n);
            cb_push_back(cb_n, 1);
        }
#endif
#ifdef RN_X8
        pack_reconfig_data_format(cb_n8);
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_n8, 1);
            pack_tile(b, cb_n8);
            cb_push_back(cb_n8, 1);
        }
#endif
#ifdef RN_ROUTER
        pack_reconfig_data_format(cb_nm);
        for (uint32_t b = 0; b < n; ++b) {
            cb_reserve_back(cb_nm, 1);
            pack_tile(b, cb_nm);
            cb_push_back(cb_nm, 1);
        }
#endif
        tile_regs_release();
    }
    cb_pop_front(cb_t, NB);

#ifdef RN_ROUTER
    // the partial logits: sum_j n_j W_j (matmul: SrcA <- in1 (W), SrcB <- in0 (n))
    cb_wait_front(cb_nm, NB);
    cb_wait_front(cb_w, NB);
    reconfig_data_format(cb_w, cb_nm);
    pack_reconfig_data_format(cb_lp);
    matmul_init(cb_nm, cb_w, 0);
    tile_regs_acquire();
    for (uint32_t j = 0; j < NB; ++j) {
        matmul_tiles(cb_nm, cb_w, j, j, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_lp, 1);
    tile_regs_wait();
    pack_tile(0, cb_lp);
    tile_regs_release();
    cb_push_back(cb_lp, 1);
    cb_pop_front(cb_nm, NB);
    if (get_absolute_logical_x() == 0) {
        // the row tile's logits: the GX partials + the bias tile (every row the bias)
        cb_wait_front(cb_lall, GX);
        cb_wait_front(cb_lb, 1);
        reconfig_data_format_srca(cb_lall);
        pack_reconfig_data_format(cb_lo);
        tile_regs_acquire();
        sum_tiles(cb_lall, GX);
        reconfig_data_format_srca(cb_lb);
        copy_init(cb_lb);
        copy_tile(cb_lb, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_lo, 1);
        tile_regs_wait();
        pack_tile(0, cb_lo);
        tile_regs_release();
        cb_push_back(cb_lo, 1);
        cb_pop_front(cb_lall, GX);
    }
#endif
    cb_pop_front(cb_r2, NB);  // (the next row tile's new residual tiles at the front)
    }
    cb_pop_front(cb_g, NB);
    cb_pop_front(cb_sc, 1);
#ifdef RN_BIAS
    cb_pop_front(cb_b, NB);
#endif
#ifdef RN_ROUTER
    cb_pop_front(cb_w, NB);
    if (get_absolute_logical_x() == 0) {
        cb_pop_front(cb_lb, 1);
    }
#endif
}
