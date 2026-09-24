// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm compute: y = (x - E[x]) * rsqrt(Var[x] + eps) * gamma + beta, x = a + b.
// Each row follows the stock layernorm.cpp sequence (fused pre-add, fp32 intermediates,
// reduce with a ones scaler then * 1/W on the SFPU, two-pass variance), so the result
// matches stock bitwise. One change: * rstd, * gamma and + beta run in DST (gamma and
// beta through dest reuse with a row broadcast), so each output tile is packed once
// instead of three fp32 pack/unpack round trips.
//
// Compile-time args: 0 Wt, 1 blk, 2 inv_w (float bits)
// Runtime args: 0 num_rows

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_SCALER = 2;
constexpr uint32_t CB_EPS = 3;
constexpr uint32_t CB_GAMMA = 4;
constexpr uint32_t CB_BETA = 5;
constexpr uint32_t CB_X = 6;       // x = a + b, then (x - E[x])^2
constexpr uint32_t CB_XMM = 7;     // x - E[x]
constexpr uint32_t CB_EX = 9;      // E[x]
constexpr uint32_t CB_EX2 = 10;    // Var[x]
constexpr uint32_t CB_EX2PE = 11;  // rsqrt(Var[x] + eps)
constexpr uint32_t CB_OUT = 16;

// cb_dst <- a + b for n tiles, blk at a time.
template <uint32_t blk>
FORCE_INLINE void pre_add(uint32_t n, uint32_t cb_dst) {
    reconfig_data_format(CB_A, CB_B);
    pack_reconfig_data_format(cb_dst);
    add_init(CB_A, CB_B);
    for (uint32_t t = 0; t < n; t += blk) {
        cb_wait_front(CB_A, blk);
        cb_wait_front(CB_B, blk);
        tile_regs_acquire();
        for (uint32_t i = 0; i < blk; ++i) {
            add_tiles(CB_A, CB_B, i, i, i);
        }
        tile_regs_commit();
        cb_pop_front(CB_A, blk);
        cb_pop_front(CB_B, blk);
        cb_reserve_back(cb_dst, blk);
        tile_regs_wait();
        for (uint32_t i = 0; i < blk; ++i) {
            pack_tile(i, cb_dst);
        }
        tile_regs_release();
        cb_push_back(cb_dst, blk);
    }
}

// cb_dst <- sum over the row of n tiles of cb_src, * 1/W. Pops cb_src when pop.
template <uint32_t inv_w, bool pop>
FORCE_INLINE void row_mean(uint32_t cb_src, uint32_t n, uint32_t cb_dst) {
    reconfig_data_format(cb_src, CB_SCALER);
    tile_regs_acquire();
    reconfig_data_format(CB_SCALER, cb_src);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_src, CB_SCALER, cb_dst);
    cb_wait_front(cb_src, n);
    for (uint32_t j = 0; j < n; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_src, CB_SCALER, j, 0, 0);
    }
    if constexpr (pop) {
        cb_pop_front(cb_src, n);
    }
    reduce_uninit();
    reconfig_data_format(cb_src, CB_SCALER);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_w);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_dst, 1);
    pack_reconfig_data_format(cb_dst);
    pack_tile(0, cb_dst);
    tile_regs_release();
    cb_push_back(cb_dst, 1);
}

// cb_xmm <- cb_x - E[x] (n tiles). Pops cb_x and CB_EX.
template <uint32_t blk>
FORCE_INLINE void center(uint32_t cb_x, uint32_t cb_xmm, uint32_t n) {
    reconfig_data_format(cb_x, CB_EX);
    pack_reconfig_data_format(cb_xmm);
    cb_wait_front(CB_EX, 1);
    cb_reserve_back(cb_xmm, n);
    sub_bcast_cols_init(cb_x, CB_EX);
    for (uint32_t t = 0; t < n; t += blk) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < blk; ++i) {
            sub_tiles_bcast_cols(cb_x, CB_EX, i, 0, i);
        }
        tile_regs_commit();
        cb_pop_front(cb_x, blk);
        tile_regs_wait();
        for (uint32_t i = 0; i < blk; ++i) {
            pack_tile(i, cb_xmm);
        }
        tile_regs_release();
        cb_push_back(cb_xmm, blk);
    }
    cb_pop_front(CB_EX, 1);
}

// cb_dst <- cb_src * cb_src for n tiles (cb_src stays).
template <uint32_t blk>
FORCE_INLINE void square(uint32_t cb_src, uint32_t cb_dst, uint32_t n) {
    reconfig_data_format(cb_src, cb_src);
    pack_reconfig_data_format(cb_dst);
    mul_init(cb_src, cb_src);
    for (uint32_t t = 0; t < n; t += blk) {
        cb_wait_front(cb_src, t + blk);
        tile_regs_acquire();
        for (uint32_t i = 0; i < blk; ++i) {
            mul_tiles(cb_src, cb_src, t + i, t + i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb_dst, blk);
        tile_regs_wait();
        for (uint32_t i = 0; i < blk; ++i) {
            pack_tile(i, cb_dst);
        }
        tile_regs_release();
        cb_push_back(cb_dst, blk);
    }
}

// CB_EX2PE <- rsqrt(CB_EX2 + eps).
FORCE_INLINE void rstd() {
    cb_wait_front(CB_EX2, 1);
    reconfig_data_format(CB_EX2, CB_EPS);
    tile_regs_acquire();
    add_init(CB_EX2, CB_EPS);
    add_tiles(CB_EX2, CB_EPS, 0, 0, 0);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(0);
    tile_regs_commit();
    cb_pop_front(CB_EX2, 1);
    cb_reserve_back(CB_EX2PE, 1);
    pack_reconfig_data_format(CB_EX2PE);
    tile_regs_wait();
    pack_tile(0, CB_EX2PE);
    tile_regs_release();
    cb_push_back(CB_EX2PE, 1);
}

// DST[idst] <- DST[idst] op row-broadcast(cb[itile]): the stock mul/add_tiles_bcast_rows
// with the DST tile as SrcA instead of a packed and unpacked fp32 tile. Mirrors
// binary_reuse_dest_init/tiles (eltwise_binary.h) with BroadcastType::ROW.
template <EltwiseBinaryType op>
FORCE_INLINE void row_bcast_reuse_init(uint32_t cb) {
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(false, false, cb)));
    MATH((llk_math_eltwise_binary_init<op, BroadcastType::ROW, MATH_FIDELITY, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        cb, cb, false)));
}

template <EltwiseBinaryType op>
FORCE_INLINE void row_bcast_reuse_tile(uint32_t cb, uint32_t itile, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, itile)));
    MATH((llk_math_eltwise_binary<
          op,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb, cb, idst, true)));
}

// CB_OUT <- CB_XMM * rstd * gamma + beta for Wt tiles, one pack per tile.
// Pops CB_XMM and CB_EX2PE.
template <uint32_t Wt, uint32_t blk>
FORCE_INLINE void normalize() {
    // The reader sends gamma/beta after the first row's inputs; they are never popped.
    cb_wait_front(CB_GAMMA, Wt);
    cb_wait_front(CB_BETA, Wt);
    cb_wait_front(CB_EX2PE, 1);
    for (uint32_t t = 0; t < Wt; t += blk) {
        reconfig_data_format(CB_XMM, CB_EX2PE);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_acquire();
        mul_bcast_cols_init(CB_XMM, CB_EX2PE);
        for (uint32_t i = 0; i < blk; ++i) {
            mul_tiles_bcast_cols(CB_XMM, CB_EX2PE, t + i, 0, i);
        }
        reconfig_data_format_srcb(CB_EX2PE, CB_GAMMA);
        row_bcast_reuse_init<EltwiseBinaryType::ELWMUL>(CB_GAMMA);
        for (uint32_t i = 0; i < blk; ++i) {
            row_bcast_reuse_tile<EltwiseBinaryType::ELWMUL>(CB_GAMMA, t + i, i);
        }
        reconfig_data_format_srcb(CB_GAMMA, CB_BETA);
        row_bcast_reuse_init<EltwiseBinaryType::ELWADD>(CB_BETA);
        for (uint32_t i = 0; i < blk; ++i) {
            row_bcast_reuse_tile<EltwiseBinaryType::ELWADD>(CB_BETA, t + i, i);
        }
        tile_regs_commit();
        cb_reserve_back(CB_OUT, blk);
        tile_regs_wait();
        for (uint32_t i = 0; i < blk; ++i) {
            pack_tile(i, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, blk);
    }
    cb_pop_front(CB_EX2PE, 1);
    cb_pop_front(CB_XMM, Wt);
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t inv_w = get_compile_time_arg_val(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(CB_A, CB_B, CB_X);
    cb_wait_front(CB_EPS, 1);
    cb_wait_front(CB_SCALER, 1);

    for (uint32_t r = 0; r < num_rows; ++r) {
        pre_add<blk>(Wt, CB_X);
        row_mean<inv_w, false>(CB_X, Wt, CB_EX);
        center<blk>(CB_X, CB_XMM, Wt);
        square<blk>(CB_XMM, CB_X, Wt);  // CB_X is empty after center
        row_mean<inv_w, true>(CB_X, Wt, CB_EX2);
        rstd();
        normalize<Wt, blk>();
    }
}
