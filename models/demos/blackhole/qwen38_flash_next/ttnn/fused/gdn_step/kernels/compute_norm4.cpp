// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Norm core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4): the l2 norm of the 4 q tiles (or the 4 k tiles) of one
// key head, so the head cores no longer run it.  l2_norm() is copied verbatim from compute_vb.cpp (same operands, same
// op order, same CB formats).  The 4 fp32 result tiles are in CB_QROW for a q core and for a k core alike (the writer,
// writer_norm4.cpp, sends them to CB_QROW or CB_KROW of the head cores).  The 4 conv tiles arrive in CB_QKV slots 0-3
// from the conv cores; the reader (reader_norm4.cpp) supplies the scaler tile in CB_SCALER.  No runtime args;
// compile-time arg 0 = VT (unused here, the shared preamble reads it).

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
#include "rowtrim4.h"

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
            rowtrim::add_unary<rowtrim::MASKED>(0, EPS_1E6);
        } else {
            rowtrim::sqrt<rowtrim::MASKED>(0);
            rowtrim::recip<rowtrim::MASKED>(0);
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
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(
        CB_QKV, CB_QKV, CB_QKV);  // bf16 for the sources and the packer, as compute_vb.cpp's startup
    cb_wait_front(CB_SCALER, 1);
    cb_wait_front(CB_QKV, HT);  // q (or k) tiles from the conv cores
    {
        FUSED_ZONE("fz_gv_n_l2_norm");
        l2_norm(0, CB_QROW);
    }
    cb_pop_front(CB_QKV, HT);
}
