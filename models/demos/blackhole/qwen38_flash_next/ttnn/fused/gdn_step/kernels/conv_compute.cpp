// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Conv core of the split GDN step (QWEN38_GDN_CONV_SPLIT): conv + SiLU of ONE tile (conv_silu() of compute.cpp with
// t = 0: the same ops in the same order on the same operand bytes), bf16 result in CB_QKV page 0 for conv_writer.cpp.

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
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

// conv[t] = silu(bf16(sum_i slot_i[t] * tap_i[t])) for the 12 q/k/v tiles; taps broadcast row 0.  The reader pushes
// the inputs per tile (CB_S tile-major: tile t's slots at 3t + i; CB_T: its taps at 4t + i), so tile t starts as soon
// as its own eight tiles landed.
ALWI void conv_silu() {
    {
        constexpr uint32_t t = 0;
        cb_wait_front(CB_P, 1);
        cb_wait_front(CB_S, 3);
        cb_wait_front(CB_T, 4);
        tile_regs_acquire();
        reconfig_data_format(CB_S, CB_T);
        mul_bcast_rows_init(CB_S, CB_T);
        for (uint32_t tap = 0; tap < 3; ++tap) {
            mul_tiles_bcast_rows(CB_S, CB_T, 3 * t + tap, 4 * t + tap, 0);
        }
        reconfig_data_format(CB_P, CB_T);
        mul_bcast_rows_init(CB_P, CB_T);
        mul_tiles_bcast_rows(CB_P, CB_T, t, 4 * t + 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_CONVSUM, 1);
        pack_reconfig_data_format(CB_CONVSUM);
        pack_tile(0, CB_CONVSUM);
        cb_push_back(CB_CONVSUM, 1);
        tile_regs_release();

        cb_wait_front(CB_CONVSUM, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_CONVSUM);
        copy_init(CB_CONVSUM);
        copy_tile(CB_CONVSUM, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_QKV, 1);
        pack_reconfig_data_format(CB_QKV);
        pack_tile(0, CB_QKV);
        cb_push_back(CB_QKV, 1);
        tile_regs_release();
        cb_pop_front(CB_CONVSUM, 1);
    }
}

}  // namespace

void kernel_main() {
#ifdef CS_NOWORK  // dispatch-cost probe: nothing to compute
    return;
#endif
    compute_kernel_hw_startup(CB_S, CB_T, CB_CONVSUM);  // as compute.cpp
    {
        FUSED_ZONE("fz_gcv_c");
        cb_wait_front(CB_P, 1);
        cb_wait_front(CB_S, 3);
        cb_wait_front(CB_T, 4);
        conv_silu();
        cb_pop_front(CB_P, 1);
        cb_pop_front(CB_S, 3);
        cb_pop_front(CB_T, 4);
    }
}
