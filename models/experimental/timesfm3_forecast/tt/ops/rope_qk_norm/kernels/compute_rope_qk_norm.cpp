// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TimesFM-3 RoPE + QK RMSNorm over q and k, one Wt-tile row (32 tokens of one head) per unit:
//   y   = x * cos + (x @ T) * sin              (T: the pair rotation, as rotary_embedding_llama)
//   out = y * rsqrt(mean(y^2) + eps) * gain    (gain: q or k)
// x @ T is exact in bf16 (T permutes with signs); y, y^2 and y * rstd stay fp32, so out is the only rounding.
// Units run in the reader's order; cos / sin change with the seq tile.
//
// Compile-time args: Wt, RH, inv_w (fp32 bits), eps (fp32 bits). Runtime: num_units, start_unit.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t RH = get_compile_time_arg_val(1);
constexpr uint32_t inv_w = get_compile_time_arg_val(2);
constexpr uint32_t eps = get_compile_time_arg_val(3);

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t cos_cb = tt::CBIndex::c_1;
constexpr uint32_t sin_cb = tt::CBIndex::c_2;
constexpr uint32_t trans_cb = tt::CBIndex::c_3;
constexpr uint32_t scaler_cb = tt::CBIndex::c_4;
constexpr uint32_t g_cb = tt::CBIndex::c_5;
constexpr uint32_t out_cb = tt::CBIndex::c_16;
constexpr uint32_t rot_cb = tt::CBIndex::c_24;
constexpr uint32_t y_cb = tt::CBIndex::c_25;
constexpr uint32_t sq_cb = tt::CBIndex::c_26;
constexpr uint32_t rstd_cb = tt::CBIndex::c_27;
constexpr uint32_t t_cb = tt::CBIndex::c_28;

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);

    CircularBuffer cb_x(x_cb);
    CircularBuffer cb_cos(cos_cb);
    CircularBuffer cb_sin(sin_cb);
    CircularBuffer cb_trans(trans_cb);
    CircularBuffer cb_scaler(scaler_cb);
    CircularBuffer cb_g(g_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_rot(rot_cb);
    CircularBuffer cb_y(y_cb);
    CircularBuffer cb_sq(sq_cb);
    CircularBuffer cb_rstd(rstd_cb);
    CircularBuffer cb_t(t_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(x_cb, trans_cb, rot_cb);
    matmul_init(x_cb, trans_cb);
    compute_kernel_hw_startup(x_cb, cos_cb, y_cb);

    cb_trans.wait_front(1);
    cb_scaler.wait_front(1);
    cb_g.wait_front(2 * Wt);

    uint32_t cur_st = 0xFFFFFFFF;
    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t st = u / (2 * RH);
        const uint32_t g0 = (u % (2 * RH)) < RH ? 0 : Wt;
        if (st != cur_st) {
            if (cur_st != 0xFFFFFFFF) {
                cb_cos.pop_front(Wt);
                cb_sin.pop_front(Wt);
            }
            cb_cos.wait_front(Wt);
            cb_sin.wait_front(Wt);
            cur_st = st;
        }
        cb_x.wait_front(Wt);

        // rot = x @ T (matmul: T in SrcA, x in SrcB)
        reconfig_data_format(trans_cb, x_cb);
        pack_reconfig_data_format(rot_cb);
        matmul_init(x_cb, trans_cb);
        cb_rot.reserve_back(Wt);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(x_cb, trans_cb, j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, rot_cb);
        }
        tile_regs_release();
        cb_rot.push_back(Wt);

        // y = x * cos + rot * sin, accumulated in DST
        cb_rot.wait_front(Wt);
        cb_y.reserve_back(Wt);
        reconfig_data_format(x_cb, cos_cb);
        pack_reconfig_data_format(y_cb);
        tile_regs_acquire();
        binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(x_cb, cos_cb, false);
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles(x_cb, cos_cb, j, j, j);
        }
        binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(rot_cb, sin_cb, true);
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles(rot_cb, sin_cb, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, y_cb);
        }
        tile_regs_release();
        cb_y.push_back(Wt);
        cb_rot.pop_front(Wt);
        cb_x.pop_front(Wt);

        // sq = y^2
        cb_y.wait_front(Wt);
        cb_sq.reserve_back(Wt);
        reconfig_data_format(y_cb, y_cb);
        pack_reconfig_data_format(sq_cb);
        mul_init(y_cb, y_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles(y_cb, y_cb, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, sq_cb);
        }
        tile_regs_release();
        cb_sq.push_back(Wt);

        // rstd = rsqrt(sum(sq) * inv_w + eps)
        cb_sq.wait_front(Wt);
        cb_rstd.reserve_back(1);
        reconfig_data_format(scaler_cb, sq_cb);
        pack_reconfig_data_format(rstd_cb);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq_cb, scaler_cb, rstd_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq_cb, scaler_cb, j, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, rstd_cb);
        tile_regs_release();
        cb_rstd.push_back(1);
        cb_sq.pop_front(Wt);

        // t = y * rstd (broadcast over columns)
        cb_rstd.wait_front(1);
        cb_t.reserve_back(Wt);
        reconfig_data_format(y_cb, rstd_cb);
        pack_reconfig_data_format(t_cb);
        mul_bcast_cols_init(y_cb, rstd_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast_cols(y_cb, rstd_cb, j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, t_cb);
        }
        tile_regs_release();
        cb_t.push_back(Wt);
        cb_rstd.pop_front(1);
        cb_y.pop_front(Wt);

        // out = t * gain (row 0 of each gain tile, broadcast over rows)
        cb_t.wait_front(Wt);
        cb_out.reserve_back(Wt);
        reconfig_data_format(t_cb, g_cb);
        pack_reconfig_data_format(out_cb);
        mul_bcast_rows_init(t_cb, g_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast_rows(t_cb, g_cb, j, g0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, out_cb);
        }
        tile_regs_release();
        cb_out.push_back(Wt);
        cb_t.pop_front(Wt);
    }
    if (cur_st != 0xFFFFFFFF) {
        cb_cos.pop_front(Wt);
        cb_sin.pop_front(Wt);
    }
}
