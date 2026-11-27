// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash drafter MLP, compute (draft_silu_mul_reader.cpp): act_t = silu(g_t) * u_t for this core's NH column
// tiles (g_t = shard tile t, u_t = shard tile NH + t): g to DST, the SFPU silu, then u multiplied in (DST -> SrcA).
// compile-time args: [cb_gu, cb_act, NH]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_gu = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t NH = get_compile_time_arg_val(2);
    constexpr uint32_t BLK = 4;

    compute_kernel_hw_startup(cb_gu, cb_gu, cb_act);
#ifndef SM_MPC
#define SM_MPC 1
#endif
    pack_reconfig_data_format(cb_act);
    for (uint32_t mi = 0; mi < SM_MPC; ++mi) {  // (row tiles per core)
        cb_wait_front(cb_gu, 2 * NH);
        for (uint32_t t0 = 0; t0 < NH; t0 += BLK) {
            const uint32_t n = t0 + BLK < NH ? BLK : NH - t0;
            tile_regs_acquire();
            copy_init(cb_gu);
            for (uint32_t b = 0; b < n; ++b) {
                copy_tile(cb_gu, t0 + b, b);
            }
            silu_tile_init();
            for (uint32_t b = 0; b < n; ++b) {
                silu_tile(b);
            }
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gu);
            for (uint32_t b = 0; b < n; ++b) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gu, NH + t0 + b, b);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < n; ++b) {
                cb_reserve_back(cb_act, 1);
                pack_tile(b, cb_act);
                cb_push_back(cb_act, 1);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_gu, 2 * NH);
    }
}
