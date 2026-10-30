// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Gate core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4): beta and decay of ONE value head, so the head cores no
// longer run the gates.  gate_scalars() is copied verbatim from compute_vb.cpp.  The reader (reader_gate4.cpp) supplies
// CB_AB (a, b tiles) and CB_DTNA (dt_bias, neg_exp_A tiles); CB_BETA and CB_DECAY (one fp32 tile each) go to the head's
// block cores (writer_gate4.cpp).  No runtime args; compile-time arg 0 = VT (unused here, the shared preamble reads
// it).

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
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_AB, CB_AB, CB_AB);  // bf16 for the sources and the packer, as compute_vb.cpp's startup
    {
        FUSED_ZONE("fz_gv_g_gates");
        gate_scalars();
    }
}
