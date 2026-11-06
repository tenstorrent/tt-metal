// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel of a conv core that also takes head items (QWEN38_GDN_LS_CONVHEAD=1, option B of
// plan_nov05/g1_gdnlane): the conv + SiLU of ONE tile (conv_compute.cpp: the same ops on the same operand bytes, in
// namespace convphase), then compute_ls.cpp (included as it is, its kernel_main renamed).  compute_kernel_hw_startup
// must run once at the start of a kernel: this file calls it once (with compute_ls's operands, all bf16 as
// conv_compute's) and turns the call inside compute_ls.cpp into a no-op with a macro.  Every op of both phases re-inits
// and re-configures its formats itself; after the conv phase the formats are restored to those of startup.  Conv CBs:
// input 0 / 1 / 2, sum 31, output 28 (CB 10 and 11 belong to the head phase here).  Runtime args: compute_ls.cpp's
// (items, s).
#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

#define compute_kernel_hw_startup(...) ((void)0)
#define kernel_main ls_head_compute_main
#include "compute_ls.cpp"
#undef kernel_main
#undef compute_kernel_hw_startup

namespace convphase {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2, CB_CONVOUT = 28, CB_CONVSUM = 31;

ALWI void conv_silu() {
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
    cb_reserve_back(CB_CONVOUT, 1);
    pack_reconfig_data_format(CB_CONVOUT);
    pack_tile(0, CB_CONVOUT);
    cb_push_back(CB_CONVOUT, 1);
    tile_regs_release();
    cb_pop_front(CB_CONVSUM, 1);
}
}  // namespace convphase

void kernel_main() {
    compute_kernel_hw_startup(
        CB_AB, CB_W, CB_OBF);  // once; the same operands compute_ls.cpp uses (CB_AB, CB_W, CB_OBF)
    {
        FUSED_ZONE("fz_gls_cc");
        convphase::conv_silu();
        cb_pop_front(convphase::CB_P, 1);
        cb_pop_front(convphase::CB_S, 3);
        cb_pop_front(convphase::CB_T, 4);
    }
    reconfig_data_format(CB_AB, CB_W);  // back to the formats of startup, then the head phase
    pack_reconfig_data_format(CB_OBF);
    ls_head_compute_main();
}
