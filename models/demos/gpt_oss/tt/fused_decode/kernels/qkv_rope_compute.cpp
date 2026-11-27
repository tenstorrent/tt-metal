// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash QKV heads split + RoPE, compute (qkv_rope_reader.cpp): per QKV tile (head h, column tile d) of a
// rotated head (h < ROPE_HEADS): out = x cos_d + (x T) sin_d (T: the rotation matrix tile, as rotary_embedding_llama's
// prefill mode); the other heads (V) copied. FP32 in DST.
// QR_HEADNORM (the drafter): x first RMS-normed per head (over its 64 columns, both tiles) and scaled by the head's
// weights (cb_g), packed in BF16 as ttnn.rms_norm's output, then rotated.
// compile-time args: [cb_x, cb_cos, cb_sin, cb_t, cb_out, HPC, ROPE_HEADS, cb_g, cb_sc, cb_sq, cb_rs, cb_tn, cb_nrm,
// inv_w bits, eps bits]

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

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t HPC = get_compile_time_arg_val(5);
    constexpr uint32_t ROPE_HEADS = get_compile_time_arg_val(6);
    constexpr uint32_t cb_g = get_compile_time_arg_val(7);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(9);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(10);
    constexpr uint32_t cb_tn = get_compile_time_arg_val(11);
    constexpr uint32_t cb_nrm = get_compile_time_arg_val(12);
    constexpr uint32_t inv_w = get_compile_time_arg_val(13);
    constexpr uint32_t eps = get_compile_time_arg_val(14);
    const uint32_t hc = get_absolute_logical_x();

    compute_kernel_hw_startup(cb_x, cb_t, cb_out);
    cb_wait_front(cb_t, 1);
#ifdef QR_HEADNORM
    cb_wait_front(cb_g, 2 * HPC);
    cb_wait_front(cb_sc, 1);
#endif
#ifndef QR_MPC
#define QR_MPC 1
#endif
    for (uint32_t mi = 0; mi < QR_MPC; ++mi) {  // (QR_MPC row tiles per core)
        cb_wait_front(cb_x, 2 * HPC);
        cb_wait_front(cb_cos, 2);
        cb_wait_front(cb_sin, 2);
        for (uint32_t p = 0; p < HPC; ++p) {
            const uint32_t h = hc * HPC + p, i0 = 2 * p;
            if (h >= ROPE_HEADS) {  // V: copied
                for (uint32_t d = 0; d < 2; ++d) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_x);
                    copy_init(cb_x);
                    copy_tile(cb_x, i0 + d, 0);
                    tile_regs_commit();
                    cb_reserve_back(cb_out, 1);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_out);
                    pack_tile(0, cb_out);
                    tile_regs_release();
                    cb_push_back(cb_out, 1);
                }
                continue;
            }
            uint32_t src = cb_x, s0 = i0;
#ifdef QR_HEADNORM
        // x^2 of both tiles (FP32)
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_x);
        mul_init(cb_x, cb_x);
        mul_tiles(cb_x, cb_x, i0, i0, 0);
        mul_tiles(cb_x, cb_x, i0 + 1, i0 + 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        for (uint32_t d = 0; d < 2; ++d) {
            cb_reserve_back(cb_sq, 1);
            pack_tile(d, cb_sq);
            cb_push_back(cb_sq, 1);
        }
        tile_regs_release();
        // rs = rsqrt(rowsum / 64 + eps) (a row-sum reduce: the scaler in SrcA, the data in SrcB)
        cb_wait_front(cb_sq, 2);
        tile_regs_acquire();
        reconfig_data_format(cb_sc, cb_sq);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, cb_rs);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, 0, 0, 0);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, 1, 0, 0);
        reduce_uninit(cb_sq);
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
        pack_reconfig_data_format(cb_rs);
        pack_tile(0, cb_rs);
        tile_regs_release();
        cb_push_back(cb_rs, 1);
        cb_pop_front(cb_sq, 2);
        // t = x rs (FP32), then n = t g (BF16, as ttnn.rms_norm's output)
        cb_wait_front(cb_rs, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_rs);
        mul_bcast_cols_init(cb_x, cb_rs);
        mul_tiles_bcast_cols(cb_x, cb_rs, i0, 0, 0);
        mul_tiles_bcast_cols(cb_x, cb_rs, i0 + 1, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_tn);
        for (uint32_t d = 0; d < 2; ++d) {
            cb_reserve_back(cb_tn, 1);
            pack_tile(d, cb_tn);
            cb_push_back(cb_tn, 1);
        }
        tile_regs_release();
        cb_pop_front(cb_rs, 1);
        cb_wait_front(cb_tn, 2);
        tile_regs_acquire();
        reconfig_data_format(cb_tn, cb_g);
        mul_bcast_rows_init(cb_tn, cb_g);
        mul_tiles_bcast_rows(cb_tn, cb_g, 0, i0, 0);
        mul_tiles_bcast_rows(cb_tn, cb_g, 1, i0 + 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_nrm);
        for (uint32_t d = 0; d < 2; ++d) {
            cb_reserve_back(cb_nrm, 1);
            pack_tile(d, cb_nrm);
            cb_push_back(cb_nrm, 1);
        }
        tile_regs_release();
        cb_pop_front(cb_tn, 2);
        cb_wait_front(cb_nrm, 2);
        src = cb_nrm;
        s0 = 0;
#endif
        for (uint32_t d = 0; d < 2; ++d) {
            tile_regs_acquire();
            // DST 0 = (x T) sin
            reconfig_data_format(cb_t, src);  // (matmul: SrcA <- in1 (T), SrcB <- in0 (x))
            matmul_init(src, cb_t, 0);
            matmul_tiles(src, cb_t, s0 + d, 0, 0);
            reconfig_data_format(cb_sin, cb_sin);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sin);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_sin, d, 0);
            // DST 1 = x cos
            reconfig_data_format(src, cb_cos);
            mul_init(src, cb_cos);
            mul_tiles(src, cb_cos, s0 + d, d, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
#ifdef QR_HEADNORM
        cb_pop_front(cb_nrm, 2);
#endif
        }
    cb_pop_front(cb_x, 2 * HPC);
    cb_pop_front(cb_cos, 2);
    cb_pop_front(cb_sin, 2);
    }
    cb_pop_front(cb_t, 1);
#ifdef QR_HEADNORM
    cb_pop_front(cb_g, 2 * HPC);
    cb_pop_front(cb_sc, 1);
#endif
}
