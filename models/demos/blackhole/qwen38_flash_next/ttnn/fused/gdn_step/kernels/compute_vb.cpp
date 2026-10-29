// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Head core of the V-split GDN step (QWEN38_GDN_CONV_SPLIT=2, stage 2): compute_cs.cpp for ONE column block of the
// head: this core owns the VT value-column tiles s*VT .. s*VT+VT-1 of the head's 128 x 128 state.  A state column never
// reads another column (v_read_j, delta_j, S[:, j], o_j), so the per-column ops are the default ones on fewer tiles
// (same operands, same accumulation order i = 0..3 in DST).  q/k (all 4 + 4 tiles) and the l2 norms are replicated on
// the VB cores.  The gated norm needs the whole o row: the four cores of a head exchange their bf16 o tiles through L1
// (the writer, writer_vb.cpp, fills CB_OBF), then every core runs the squares and the reduce over all 4 tiles in the
// default order (a different add order would change the fp32 result) and finishes its own VT tiles. Compile-time args:
// 0 VT.  Runtime args: 0 items, 1 s (the block).  Defines: DECAY_FIRST (the decay of the state runs before the wait for
// the conv tiles).

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
constexpr uint32_t CB_OMINE = 29, CB_SQO = 30;  // this core's o columns (bf16); the squares of the whole o row (fp32,
                                                // as compute.cpp's CB_SQO = CB_DELTAB)
// phase-E reuse of phase-D buffers (same format; the value-column tiles are pushed in units of VT, which divides the
// depth 4)
constexpr uint32_t CB_OBF = CB_CONVSUM, CB_NRMW = CB_SQ, CB_NRM = CB_UNIT, CB_SIG = CB_VREAD;

constexpr uint32_t HT = 4;                            // 128 / 32
constexpr uint32_t QKV_TILES = 12;                    // q, k, v x HT
constexpr uint32_t VT = get_compile_time_arg_val(0);  // value-column tiles of this core (4 / VB)
constexpr uint32_t ST = HT * VT;                      // its state tiles
constexpr uint32_t EPS_1E6 = 0x358637BD;              // 1e-6f: l2 norm eps and RMS_NORM_EPS
constexpr uint32_t QK_SCALE = 0x3DB504F3;             // 128^-0.5
constexpr uint32_t F_ONE = 0x3F800000, F_TWENTY = 0x41A00000;

#ifdef DEBUG_TAPS
// pack DST tile idst a second time into the debug CB; restore_cb is the CB the surrounding code packs into
ALWI void tap(uint32_t idst, uint32_t restore_cb) {
    cb_reserve_back(CB_DBG, 1);
    pack_reconfig_data_format(CB_DBG);
    pack_tile(idst, CB_DBG);
    cb_push_back(CB_DBG, 1);
    pack_reconfig_data_format(restore_cb);
}
#endif

// unit = x * rsqrt(sum(x*x) + eps) in the oracle's bf16 chain; the fp32 copy of unit -> out_cb.
ALWI void l2_norm(uint32_t base, uint32_t out_cb) {
    reconfig_data_format(CB_QKV, CB_QKV);
    mul_init(CB_QKV, CB_QKV);
    pack_reconfig_data_format(CB_SQ);
    cb_reserve_back(CB_SQ, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        mul_tiles(CB_QKV, CB_QKV, base + t, base + t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQ);
        tile_regs_release();
    }
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
#ifdef DEBUG_TAPS
    if (base == 0) {
        tap(0, CB_STAT);
    }
#endif
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
#ifdef DEBUG_TAPS
        if (base == 0) {
            tap(0, CB_STAT);
        }
#endif
        tile_regs_release();
        cb_pop_front(CB_STAT, 1);
    }

    cb_wait_front(CB_STAT, 1);
    reconfig_data_format(CB_QKV, CB_STAT);
    mul_bcast_cols_init(CB_QKV, CB_STAT);
    pack_reconfig_data_format(CB_UNIT);
    cb_reserve_back(CB_UNIT, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(CB_QKV, CB_STAT, base + t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_UNIT);
        tile_regs_release();
    }
    cb_push_back(CB_UNIT, HT);
    cb_pop_front(CB_STAT, 1);

    cb_wait_front(CB_UNIT, HT);
    reconfig_data_format_srca(CB_UNIT);
    copy_init(CB_UNIT);
    pack_reconfig_data_format(out_cb);
    cb_reserve_back(out_cb, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        copy_tile(CB_UNIT, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
#ifdef DEBUG_TAPS
        if (base == 0 && t == 0) {
            tap(0, out_cb);
        }
#endif
        tile_regs_release();
    }
    cb_push_back(out_cb, HT);
    cb_pop_front(CB_UNIT, HT);
}

// beta = bf16(sigmoid(b)) and decay = exp(neg_exp_A * softplus(a + dt_bias)) as full fp32 tiles.
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
#ifdef DEBUG_TAPS
    tap(0, CB_BETA);
#endif
    tile_regs_release();
    cb_pop_front(CB_STAT, 1);

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
#ifdef DEBUG_TAPS
    tap(0, CB_DECAY);
#endif
    tile_regs_release();
    cb_pop_front(CB_AB, 2);
    cb_pop_front(CB_DTNA, 2);
}

// S' = S * decay (this core's ST state tiles).
ALWI void decay_state() {
    cb_wait_front(CB_STATE, ST);
    cb_wait_front(CB_DECAY, 1);
    reconfig_data_format_srca(CB_STATE);
    copy_init(CB_STATE);
    mul_binary_tile_init();
    pack_reconfig_data_format(CB_SDEC);
    cb_reserve_back(CB_SDEC, ST);
    cb_reserve_back(CB_SDECC, ST);
    for (uint32_t t = 0; t < ST; t += 2) {
        tile_regs_acquire();
        copy_tile(CB_STATE, t, 0);
        copy_tile(CB_DECAY, 0, 1);
        mul_binary_tile(0, 1, 0);
        copy_tile(CB_STATE, t + 1, 2);
        copy_tile(CB_DECAY, 0, 3);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SDEC);
        pack_tile(2, CB_SDEC);
        pack_tile(0, CB_SDECC);
        pack_tile(2, CB_SDECC);
        tile_regs_release();
    }
    cb_push_back(CB_SDEC, ST);
    cb_push_back(CB_SDECC, ST);
    cb_wait_front(CB_SDEC, ST);
    cb_wait_front(CB_SDECC, ST);
}

// v_read = k S'; delta_b = ((v - v_read) * mask) * beta; S_new = S' + k^T delta_b; o = (q S_new) * 128^-0.5, all on
// this core's VT columns.  o goes to CB_OMINE (bf16) for the exchange.
ALWI void recurrence_rest() {
    cb_wait_front(CB_BETA, 1);
    cb_wait_front(CB_MASK, 1);
    cb_wait_front(CB_KROW, HT);
    cb_wait_front(CB_QROW, HT);
    cb_wait_front(CB_QKV, VT);  // this core's v tiles

    reconfig_data_format<SrcOrder::Reverse>(CB_KROW, CB_SDEC);
    matmul_init(CB_KROW, CB_SDEC);
    pack_reconfig_data_format(CB_VREAD);
    tile_regs_acquire();
    for (uint32_t j = 0; j < VT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_KROW, CB_SDEC, i, i * VT + j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_VREAD, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        pack_tile(j, CB_VREAD);
    }
    cb_push_back(CB_VREAD, VT);
    tile_regs_release();

    cb_wait_front(CB_VREAD, VT);
    pack_reconfig_data_format(CB_DELTAB);
    cb_reserve_back(CB_DELTAB, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_QKV);
        copy_init(CB_QKV);
        copy_tile(CB_QKV, j, 0);
        reconfig_data_format_srca(CB_VREAD);
        copy_init(CB_VREAD);
        copy_tile(CB_VREAD, j, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_MASK);
        copy_init(CB_MASK);
        copy_tile(CB_MASK, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_BETA);
        copy_init(CB_BETA);
        copy_tile(CB_BETA, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_DELTAB);
        tile_regs_release();
    }
    cb_push_back(CB_DELTAB, VT);
    cb_pop_front(CB_VREAD, VT);
    cb_pop_front(CB_QKV, VT);

    reconfig_data_format_srca(CB_KROW);
    transpose_init(CB_KROW);
    pack_reconfig_data_format(CB_KCOL);
    cb_reserve_back(CB_KCOL, HT);
    for (uint32_t i = 0; i < HT; ++i) {
        tile_regs_acquire();
        transpose_tile(CB_KROW, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_KCOL);
        tile_regs_release();
    }
    cb_push_back(CB_KCOL, HT);
    cb_pop_front(CB_KROW, HT);

    cb_wait_front(CB_KCOL, HT);
    cb_wait_front(CB_DELTAB, VT);
    cb_reserve_back(CB_OUTS, ST);  // the writer has drained the previous item's state
    cb_reserve_back(CB_SNEW, ST);
    pack_reconfig_data_format(CB_SNEW);
    for (uint32_t i = 0; i < HT; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_SDECC);
        copy_init(CB_SDECC);
        for (uint32_t j = 0; j < VT; ++j) {
            copy_tile(CB_SDECC, i * VT + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(CB_KCOL, CB_DELTAB);
        matmul_init(CB_KCOL, CB_DELTAB);
        for (uint32_t j = 0; j < VT; ++j) {
            matmul_tiles(CB_KCOL, CB_DELTAB, i, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < VT; ++j) {
            pack_tile(j, CB_SNEW);
        }
        tile_regs_release();
    }
    cb_push_back(CB_SNEW, ST);
    cb_pop_front(CB_SDEC, ST);
    cb_pop_front(CB_SDECC, ST);
    cb_pop_front(CB_KCOL, HT);
    cb_pop_front(CB_DELTAB, VT);
    cb_pop_front(CB_STATE, ST);
    cb_pop_front(CB_DECAY, 1);
    cb_pop_front(CB_BETA, 1);
    cb_pop_front(CB_MASK, 1);

    cb_wait_front(CB_SNEW, ST);
    reconfig_data_format<SrcOrder::Reverse>(CB_QROW, CB_SNEW);
    matmul_init(CB_QROW, CB_SNEW);
    pack_reconfig_data_format(CB_OMINE);
    tile_regs_acquire();
    for (uint32_t j = 0; j < VT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_QROW, CB_SNEW, i, i * VT + j, j);
        }
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < VT; ++j) {
        mul_unary_tile(j, QK_SCALE);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_OMINE, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        pack_tile(j, CB_OMINE);
    }
    cb_push_back(CB_OMINE, VT);  // the writer sends it to the head's block cores first ...
    tile_regs_release();
    cb_pop_front(CB_SNEW, ST);
    cb_push_back(CB_OUTS, ST);  // ... then drains the state (same bytes as CB_SNEW)
    cb_pop_front(CB_QROW, HT);
}

// gated = bf16(bf16(w * bf16(o * rsqrt(mean(o^2) + eps))) * sigmoid(z)) for this core's VT tiles; CB_OBF holds the
// whole o row (4 tiles, filled by the writer's exchange: tile s*VT + c from the core of block s).
ALWI void gated_norm(uint32_t s) {
    cb_wait_front(CB_OBF, HT);
    cb_wait_front(CB_W, VT);
    cb_wait_front(CB_Z, VT);

    reconfig_data_format(CB_OBF, CB_OBF);
    mul_init(CB_OBF, CB_OBF);
    pack_reconfig_data_format(CB_SQO);
    cb_reserve_back(CB_SQO, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        mul_tiles(CB_OBF, CB_OBF, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQO);
        tile_regs_release();
    }
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
    cb_reserve_back(CB_NRM, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(CB_OBF, CB_RS, s * VT + j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_NRM);
        tile_regs_release();
    }
    cb_push_back(CB_NRM, VT);
    cb_pop_front(CB_RS, 1);
    cb_pop_front(CB_OBF, HT);

    cb_wait_front(CB_NRM, VT);
    reconfig_data_format(CB_NRM, CB_W);
    mul_bcast_rows_init(CB_NRM, CB_W);
    pack_reconfig_data_format(CB_NRMW);
    cb_reserve_back(CB_NRMW, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(CB_NRM, CB_W, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_NRMW);
        tile_regs_release();
    }
    cb_push_back(CB_NRMW, VT);
    cb_pop_front(CB_NRM, VT);

    reconfig_data_format_srca(CB_Z);
    copy_init(CB_Z);
    sigmoid_tile_init<false>();
    pack_reconfig_data_format(CB_SIG);
    cb_reserve_back(CB_SIG, VT);
    for (uint32_t j = 0; j < VT; ++j) {
        tile_regs_acquire();
        copy_tile(CB_Z, j, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SIG);
        tile_regs_release();
    }
    cb_push_back(CB_SIG, VT);
    cb_pop_front(CB_Z, VT);

    cb_wait_front(CB_NRMW, VT);
    cb_wait_front(CB_SIG, VT);
    pack_reconfig_data_format(CB_OUTG);
    cb_reserve_back(CB_OUTG, VT);
    mul_binary_tile_init();
    for (uint32_t j = 0; j < VT; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_NRMW);
        copy_init(CB_NRMW);
        copy_tile(CB_NRMW, j, 0);
        reconfig_data_format_srca(CB_SIG);
        copy_init(CB_SIG);
        copy_tile(CB_SIG, j, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUTG);
        tile_regs_release();
    }
    cb_push_back(CB_OUTG, VT);
    cb_pop_front(CB_NRMW, VT);
    cb_pop_front(CB_SIG, VT);
}
}  // namespace

void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(CB_AB, CB_W, CB_OBF);  // formats as compute.cpp's CB_S / CB_T / CB_CONVSUM (bf16)
    cb_wait_front(CB_SCALER, 2);

    for (uint32_t item = 0; item < items; ++item) {
        {
            FUSED_ZONE("fz_gv_c_gates");
            gate_scalars();
        }
#ifdef DECAY_FIRST
        {
            FUSED_ZONE("fz_gv_c_decay");
            decay_state();
        }
#endif
        cb_wait_front(CB_QKV, 2 * HT);  // q and k tiles from the conv cores
        {
            FUSED_ZONE("fz_gv_c_l2_norms");
            l2_norm(0, CB_QROW);
            l2_norm(HT, CB_KROW);
        }
        cb_pop_front(CB_QKV, 2 * HT);
#ifndef DECAY_FIRST
        {
            FUSED_ZONE("fz_gv_c_decay");
            decay_state();
        }
#endif
        {
            FUSED_ZONE("fz_gv_c_recurrence");
            recurrence_rest();
        }
        {
            FUSED_ZONE("fz_gv_c_gated_norm");
            gated_norm(s);
        }
    }
}
