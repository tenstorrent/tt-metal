// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One GDN decode step for one (lane, value head) work item per loop: conv + SiLU, q/k l2 norm, the decay and write
// gates, the fp32 delta-rule state update, the read-out, the gated RMSNorm.  Rounding points follow tt/gdn.py (the
// oracle): fp32 conv sum -> bf16 -> silu -> bf16; the l2 norm's bf16 chain; beta in bf16; the state in fp32 (SFPU
// multiply / fp32 accumulate); matmul operands at HiFi4.  DST holds fp32 (4 tiles per acquire).
// Runtime args: 0 items on this core.  Define DEBUG_TAPS to also emit conv, decay/beta and o to CB 31.

#ifdef DEBUG_TAPS
// the debug-tap build keeps the reference kernel's exact tap order: use it unchanged
#include "compute.cpp"
#else

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "../../kernels/source_softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "../../kernels/source_reciprocal.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2, CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_CONVSUM = 10, CB_QKV = 11, CB_SQ = 12, CB_STAT = 13, CB_UNIT = 14;
constexpr uint32_t CB_QROW = 15, CB_KROW = 16, CB_KCOL = 17, CB_BETA = 18, CB_DECAY = 19, CB_SDEC = 20, CB_SDECC = 21;
constexpr uint32_t CB_VREAD = 22, CB_DELTAB = 23, CB_SNEW = 24, CB_OUTS = 25, CB_OUTG = 26, CB_RS = 27, CB_DBG = 28;
// phase-E reuse of phase-D buffers (same format and depth)
constexpr uint32_t CB_OBF = CB_CONVSUM, CB_NRMW = CB_SQ, CB_NRM = CB_UNIT, CB_SIG = CB_VREAD, CB_SQO = CB_DELTAB;

constexpr uint32_t HT = 4;          // 128 / 32
constexpr uint32_t QKV_TILES = 12;  // q, k, v x HT
constexpr uint32_t STATE_TILES = 16;
constexpr uint32_t EPS_1E6 = 0x358637BD;   // 1e-6f: l2 norm eps and RMS_NORM_EPS
constexpr uint32_t QK_SCALE = 0x3DB504F3;  // 128^-0.5
constexpr uint32_t F_ONE = 0x3F800000, F_TWENTY = 0x41A00000;

// conv[t] = silu(bf16(sum_i slot_i[t] * tap_i[t])) for the 12 q/k/v tiles; taps broadcast row 0.  v2: CONV_BATCH tiles
// per DST acquire (tile d of the batch in DST d, each fresh DST tile accumulates its four products as before), the
// P and S operands share one mul_bcast_rows init (same bf16 tiles, same shapes), the silu stage runs per batch with one
// copy/silu init.  Per-tile arithmetic order is unchanged.  The reader still pushes per tile: the waits stay per tile.
constexpr uint32_t CONV_BATCH = 4;
ALWI void conv_silu() {
    pack_reconfig_data_format(CB_CONVSUM);  // CB_CONVSUM and CB_QKV are both bf16 tiles: one pack config serves both
    for (uint32_t b = 0; b < QKV_TILES; b += CONV_BATCH) {
        cb_reserve_back(CB_CONVSUM, CONV_BATCH);
        tile_regs_acquire();
        reconfig_data_format(CB_S, CB_T);
        mul_bcast_rows_init(CB_S, CB_T);
        for (uint32_t d = 0; d < CONV_BATCH; ++d) {
            const uint32_t t = b + d;
            cb_wait_front(CB_P, t + 1);
            cb_wait_front(CB_S, 3 * (t + 1));
            cb_wait_front(CB_T, 4 * (t + 1));
            for (uint32_t tap = 0; tap < 3; ++tap) {
                mul_tiles_bcast_rows(CB_S, CB_T, 3 * t + tap, 4 * t + tap, d);
            }
            mul_tiles_bcast_rows(CB_P, CB_T, t, 4 * t + 3, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < CONV_BATCH; ++d) {
            pack_tile(d, CB_CONVSUM);
        }
        cb_push_back(CB_CONVSUM, CONV_BATCH);
        tile_regs_release();

        cb_wait_front(CB_CONVSUM, CONV_BATCH);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_CONVSUM);
        copy_init(CB_CONVSUM);
        for (uint32_t d = 0; d < CONV_BATCH; ++d) {
            copy_tile(CB_CONVSUM, d, d);
        }
        silu_tile_init();
        for (uint32_t d = 0; d < CONV_BATCH; ++d) {
            silu_tile(d);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_QKV, CONV_BATCH);
        for (uint32_t d = 0; d < CONV_BATCH; ++d) {
            pack_tile(d, CB_QKV);
        }
        cb_push_back(CB_QKV, CONV_BATCH);
        tile_regs_release();
        cb_pop_front(CB_CONVSUM, CONV_BATCH);
    }
}

// unit = x * rsqrt(sum(x*x) + eps) in the oracle's bf16 chain; the fp32 copy of unit -> out_cb.  v2: the four-tile
// loops share one DST acquire (tile t in DST t).
ALWI void l2_norm(uint32_t base, uint32_t out_cb) {
    reconfig_data_format(CB_QKV, CB_QKV);
    mul_init(CB_QKV, CB_QKV);
    pack_reconfig_data_format(CB_SQ);
    cb_reserve_back(CB_SQ, HT);
    tile_regs_acquire();
    for (uint32_t t = 0; t < HT; ++t) {
        mul_tiles(CB_QKV, CB_QKV, base + t, base + t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < HT; ++t) {
        pack_tile(t, CB_SQ);
    }
    tile_regs_release();
    cb_push_back(CB_SQ, HT);

    cb_wait_front(CB_SQ, HT);
    reconfig_data_format(CB_SCALER, CB_SQ);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_STAT);
    pack_reconfig_data_format(CB_STAT);
    tile_regs_acquire();
    for (uint32_t t = 0; t < HT; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_STAT, 1);
    pack_tile(0, CB_STAT);
    cb_push_back(CB_STAT, 1);
    tile_regs_release();
    reduce_uninit();
    cb_pop_front(CB_SQ, HT);

    // + eps -> bf16, then rsqrt -> bf16
    for (uint32_t step = 0; step < 2; ++step) {
        cb_wait_front(CB_STAT, 1);
        reconfig_data_format_srca(CB_STAT);
        copy_init(CB_STAT);
        if (step == 0) {
            binop_with_scalar_tile_init();
        } else {
            sqrt_tile_init();
            qwen38_recip_tile_init();
        }
        tile_regs_acquire();
        copy_tile(CB_STAT, 0, 0);
        if (step == 0) {
            add_unary_tile(0, EPS_1E6);
        } else {
            sqrt_tile(0);
            qwen38_recip_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_STAT, 1);
        pack_tile(0, CB_STAT);
        cb_push_back(CB_STAT, 1);
        tile_regs_release();
        cb_pop_front(CB_STAT, 1);
    }

    cb_wait_front(CB_STAT, 1);
    reconfig_data_format(CB_QKV, CB_STAT);
    mul_bcast_cols_init(CB_QKV, CB_STAT);
    pack_reconfig_data_format(CB_UNIT);
    cb_reserve_back(CB_UNIT, HT);
    tile_regs_acquire();
    for (uint32_t t = 0; t < HT; ++t) {
        mul_tiles_bcast_cols(CB_QKV, CB_STAT, base + t, 0, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < HT; ++t) {
        pack_tile(t, CB_UNIT);
    }
    tile_regs_release();
    cb_push_back(CB_UNIT, HT);
    cb_pop_front(CB_STAT, 1);

    cb_wait_front(CB_UNIT, HT);
    reconfig_data_format_srca(CB_UNIT);
    copy_init(CB_UNIT);
    pack_reconfig_data_format(out_cb);
    cb_reserve_back(out_cb, HT);
    tile_regs_acquire();
    for (uint32_t t = 0; t < HT; ++t) {
        copy_tile(CB_UNIT, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < HT; ++t) {
        pack_tile(t, out_cb);
    }
    tile_regs_release();
    cb_push_back(out_cb, HT);
    cb_pop_front(CB_UNIT, HT);
}

// beta = bf16(sigmoid(b)) and decay = exp(neg_exp_A * softplus(a + dt_bias)) as full fp32 tiles.  v2: the decay chain
// runs between the sigmoid's pack and the bf16 -> fp32 beta copy, which hides the pack -> unpack round trip (the two
// chains share no state).
ALWI void gate_scalars() {
    cb_wait_front(CB_AB, 2);
    cb_wait_front(CB_DTNA, 2);

    tile_regs_acquire();
    reconfig_data_format_srca(CB_AB);
    unary_bcast_init<BroadcastType::SCALAR>(CB_AB);
    unary_bcast<BroadcastType::SCALAR>(CB_AB, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<VectorMode::RC, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_STAT, 1);
    pack_reconfig_data_format(CB_STAT);
    pack_tile(0, CB_STAT);
    cb_push_back(CB_STAT, 1);
    tile_regs_release();

    tile_regs_acquire();
    reconfig_data_format_srca(CB_AB);
    unary_bcast_init<BroadcastType::SCALAR>(CB_AB);
    unary_bcast<BroadcastType::SCALAR>(CB_AB, 0, 0);
    reconfig_data_format_srca(CB_DTNA);
    copy_init(CB_DTNA);
    copy_tile(CB_DTNA, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    qwen38_softplus_tile(0, F_ONE, F_ONE, F_TWENTY);
    copy_tile(CB_DTNA, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_DECAY, 1);
    pack_reconfig_data_format(CB_DECAY);
    pack_tile(0, CB_DECAY);
    cb_push_back(CB_DECAY, 1);
    tile_regs_release();
    cb_pop_front(CB_AB, 2);
    cb_pop_front(CB_DTNA, 2);

    cb_wait_front(CB_STAT, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_STAT);
    copy_init(CB_STAT);
    copy_tile(CB_STAT, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_BETA, 1);
    pack_reconfig_data_format(CB_BETA);
    pack_tile(0, CB_BETA);
    cb_push_back(CB_BETA, 1);
    tile_regs_release();
    cb_pop_front(CB_STAT, 1);
}

// S' = S * decay; v_read = k S'; delta_b = ((v - v_read) * mask) * beta; S_new = S' + k^T delta_b; o = (q S_new) *
// 128^-0.5.
ALWI void recurrence() {
    cb_wait_front(CB_STATE, STATE_TILES);
    cb_wait_front(CB_DECAY, 1);
    cb_wait_front(CB_BETA, 1);
    cb_wait_front(CB_MASK, 1);
    cb_wait_front(CB_KROW, HT);
    cb_wait_front(CB_QROW, HT);
    cb_wait_front(CB_QKV, HT);  // v

    reconfig_data_format_srca(CB_STATE);
    copy_init(CB_STATE);
    mul_binary_tile_init();
    pack_reconfig_data_format(CB_SDEC);
    cb_reserve_back(CB_SDEC, STATE_TILES);
    cb_reserve_back(CB_SDECC, STATE_TILES);
    // v2: decay sits in DST 3 for a group of up to three state tiles (the multiply reads it, never writes it)
    for (uint32_t t = 0; t < STATE_TILES; t += 3) {
        const uint32_t n = (STATE_TILES - t) < 3 ? (STATE_TILES - t) : 3;
        tile_regs_acquire();
        copy_tile(CB_DECAY, 0, 3);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_STATE, t + i, i);
            mul_binary_tile(i, 3, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, CB_SDEC);
        }
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, CB_SDECC);
        }
        tile_regs_release();
    }
    cb_push_back(CB_SDEC, STATE_TILES);
    cb_push_back(CB_SDECC, STATE_TILES);
    cb_wait_front(CB_SDEC, STATE_TILES);
    cb_wait_front(CB_SDECC, STATE_TILES);

    reconfig_data_format<SrcOrder::Reverse>(CB_KROW, CB_SDEC);
    matmul_init(CB_KROW, CB_SDEC);
    pack_reconfig_data_format(CB_VREAD);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_KROW, CB_SDEC, i, i * HT + j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_VREAD, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_VREAD);
    }
    cb_push_back(CB_VREAD, HT);
    tile_regs_release();

    cb_wait_front(CB_VREAD, HT);
    pack_reconfig_data_format(CB_DELTAB);
    cb_reserve_back(CB_DELTAB, HT);
    // v2: two heads of the delta (j, j + 1) per acquire in DST 0/2; the mask and beta tiles are loaded once per pair
    for (uint32_t j = 0; j < HT; j += 2) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_QKV);
        copy_init(CB_QKV);
        copy_tile(CB_QKV, j, 0);
        copy_tile(CB_QKV, j + 1, 2);
        reconfig_data_format_srca(CB_VREAD);
        copy_init(CB_VREAD);
        copy_tile(CB_VREAD, j, 1);
        copy_tile(CB_VREAD, j + 1, 3);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        sub_binary_tile(2, 3, 2);
        reconfig_data_format_srca(CB_MASK);
        copy_init(CB_MASK);
        copy_tile(CB_MASK, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 1, 2);
        reconfig_data_format_srca(CB_BETA);
        copy_init(CB_BETA);
        copy_tile(CB_BETA, 0, 1);
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_DELTAB);
        pack_tile(2, CB_DELTAB);
        tile_regs_release();
    }
    cb_push_back(CB_DELTAB, HT);
    cb_pop_front(CB_VREAD, HT);
    cb_pop_front(CB_QKV, HT);

    reconfig_data_format_srca(CB_KROW);
    transpose_init(CB_KROW);
    pack_reconfig_data_format(CB_KCOL);
    cb_reserve_back(CB_KCOL, HT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < HT; ++i) {
        transpose_tile(CB_KROW, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < HT; ++i) {
        pack_tile(i, CB_KCOL);
    }
    tile_regs_release();
    cb_push_back(CB_KCOL, HT);
    cb_pop_front(CB_KROW, HT);

    cb_wait_front(CB_KCOL, HT);
    cb_wait_front(CB_DELTAB, HT);
    cb_reserve_back(CB_OUTS, STATE_TILES);  // the writer has drained the previous item's state
    cb_reserve_back(CB_SNEW, STATE_TILES);
    pack_reconfig_data_format(CB_SNEW);
    for (uint32_t i = 0; i < HT; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_SDECC);
        copy_init(CB_SDECC);
        for (uint32_t j = 0; j < HT; ++j) {
            copy_tile(CB_SDECC, i * HT + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(CB_KCOL, CB_DELTAB);
        matmul_init(CB_KCOL, CB_DELTAB);
        for (uint32_t j = 0; j < HT; ++j) {
            matmul_tiles(CB_KCOL, CB_DELTAB, i, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < HT; ++j) {
            pack_tile(j, CB_SNEW);
        }
        tile_regs_release();
    }
    cb_push_back(CB_SNEW, STATE_TILES);
    cb_pop_front(CB_SDEC, STATE_TILES);
    cb_pop_front(CB_SDECC, STATE_TILES);
    cb_pop_front(CB_KCOL, HT);
    cb_pop_front(CB_DELTAB, HT);
    cb_pop_front(CB_STATE, STATE_TILES);
    cb_pop_front(CB_DECAY, 1);
    cb_pop_front(CB_BETA, 1);
    cb_pop_front(CB_MASK, 1);

    cb_wait_front(CB_SNEW, STATE_TILES);
    reconfig_data_format<SrcOrder::Reverse>(CB_QROW, CB_SNEW);
    matmul_init(CB_QROW, CB_SNEW);
    pack_reconfig_data_format(CB_OBF);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_QROW, CB_SNEW, i, i * HT + j, j);
        }
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < HT; ++j) {
        mul_unary_tile(j, QK_SCALE);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_OBF, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_OBF);
    }
    cb_push_back(CB_OBF, HT);
    tile_regs_release();
    cb_pop_front(CB_SNEW, STATE_TILES);
    cb_push_back(CB_OUTS, STATE_TILES);  // same bytes as CB_SNEW: hands the new state to the writer
    cb_pop_front(CB_QROW, HT);
}

// gated = bf16(bf16(w * bf16(o * rsqrt(mean(o^2) + eps))) * sigmoid(z)).  v2: four-tile loops share one DST acquire;
// the final product loads two (norm, sigmoid) pairs per acquire.
ALWI void gated_norm() {
    cb_wait_front(CB_OBF, HT);
    cb_wait_front(CB_W, HT);
    cb_wait_front(CB_Z, HT);

    reconfig_data_format(CB_OBF, CB_OBF);
    mul_init(CB_OBF, CB_OBF);
    pack_reconfig_data_format(CB_SQO);
    cb_reserve_back(CB_SQO, HT);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        mul_tiles(CB_OBF, CB_OBF, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_SQO);
    }
    tile_regs_release();
    cb_push_back(CB_SQO, HT);

    cb_wait_front(CB_SQO, HT);
    reconfig_data_format(CB_SCALER, CB_SQO);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQO, CB_SCALER, CB_RS);
    pack_reconfig_data_format(CB_RS);
    binop_with_scalar_tile_init();
    sqrt_tile_init();
    qwen38_recip_tile_init();
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQO, CB_SCALER, j, 1, 0);
    }
    add_unary_tile(0, EPS_1E6);
    sqrt_tile(0);
    qwen38_recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_RS, 1);
    pack_tile(0, CB_RS);
    cb_push_back(CB_RS, 1);
    tile_regs_release();
    reduce_uninit();
    cb_pop_front(CB_SQO, HT);

    cb_wait_front(CB_RS, 1);
    reconfig_data_format(CB_OBF, CB_RS);
    mul_bcast_cols_init(CB_OBF, CB_RS);
    pack_reconfig_data_format(CB_NRM);
    cb_reserve_back(CB_NRM, HT);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        mul_tiles_bcast_cols(CB_OBF, CB_RS, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_NRM);
    }
    tile_regs_release();
    cb_push_back(CB_NRM, HT);
    cb_pop_front(CB_RS, 1);
    cb_pop_front(CB_OBF, HT);

    cb_wait_front(CB_NRM, HT);
    reconfig_data_format(CB_NRM, CB_W);
    mul_bcast_rows_init(CB_NRM, CB_W);
    pack_reconfig_data_format(CB_NRMW);
    cb_reserve_back(CB_NRMW, HT);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        mul_tiles_bcast_rows(CB_NRM, CB_W, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_NRMW);
    }
    tile_regs_release();
    cb_push_back(CB_NRMW, HT);
    cb_pop_front(CB_NRM, HT);

    reconfig_data_format_srca(CB_Z);
    copy_init(CB_Z);
    sigmoid_tile_init<false>();
    pack_reconfig_data_format(CB_SIG);
    cb_reserve_back(CB_SIG, HT);
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        copy_tile(CB_Z, j, j);
    }
    for (uint32_t j = 0; j < HT; ++j) {
        sigmoid_tile<VectorMode::RC, false>(j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < HT; ++j) {
        pack_tile(j, CB_SIG);
    }
    tile_regs_release();
    cb_push_back(CB_SIG, HT);
    cb_pop_front(CB_Z, HT);

    cb_wait_front(CB_NRMW, HT);
    cb_wait_front(CB_SIG, HT);
    pack_reconfig_data_format(CB_OUTG);
    cb_reserve_back(CB_OUTG, HT);
    mul_binary_tile_init();
    for (uint32_t j = 0; j < HT; j += 2) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_NRMW);
        copy_init(CB_NRMW);
        copy_tile(CB_NRMW, j, 0);
        copy_tile(CB_NRMW, j + 1, 2);
        reconfig_data_format_srca(CB_SIG);
        copy_init(CB_SIG);
        copy_tile(CB_SIG, j, 1);
        copy_tile(CB_SIG, j + 1, 3);
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUTG);
        pack_tile(2, CB_OUTG);
        tile_regs_release();
    }
    cb_push_back(CB_OUTG, HT);
    cb_pop_front(CB_NRMW, HT);
    cb_pop_front(CB_SIG, HT);
}
}  // namespace

void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_S, CB_T, CB_CONVSUM);
    cb_wait_front(CB_SCALER, 2);  // reader-built: tile 0 = 1.0 (sums), tile 1 = 1/128 (means)

    for (uint32_t item = 0; item < items; ++item) {
        {
            FUSED_ZONE("fz_gs_c_wait");
            cb_wait_front(CB_P, 1);  // the first tile's group; the conv waits for the rest tile by tile
            cb_wait_front(CB_S, 3);
            cb_wait_front(CB_T, 4);
        }
        {
            FUSED_ZONE("fz_gs_c_conv_silu");
            conv_silu();
        }
        cb_pop_front(CB_P, QKV_TILES);
        cb_pop_front(CB_S, 3 * QKV_TILES);
        cb_pop_front(CB_T, 4 * QKV_TILES);

        cb_wait_front(CB_QKV, 2 * HT);
        {
            FUSED_ZONE("fz_gs_c_l2_norms");
            l2_norm(0, CB_QROW);
            l2_norm(HT, CB_KROW);
        }
        cb_pop_front(CB_QKV, 2 * HT);

        {
            FUSED_ZONE("fz_gs_c_gates");
            gate_scalars();
        }
        {
            FUSED_ZONE("fz_gs_c_recurrence");
            recurrence();
        }
        {
            FUSED_ZONE("fz_gs_c_gated_norm");
            gated_norm();
        }
    }
}

#endif  // DEBUG_TAPS
