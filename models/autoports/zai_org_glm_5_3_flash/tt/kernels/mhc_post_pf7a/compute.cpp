// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7a prefill post-mix — compute (common.hpp): per unit and output stream s, ONE DST pass: y, x_0..3, K_s, L_s
// copied into DST (fp32 unpack-to-dest: exact), then mix_sfpu.hpp's mpf7_mix_tile writes
// out_s = post_s * y + sum_i C[i][s] x_i into DST tile 7 (fp32 FMAs). Full-sync fp32 DST (8 tiles).
// Compile-time: CT. Runtime: first unit, unit count.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

#include "common.hpp"
#include "mix_sfpu.hpp"

using namespace mpf7;

namespace {

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);  // bf16 y / fp32 residual + coefficients alternate
    copy_init(cb);
    copy_tile(cb, i, d);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_X, CB_OUT);
    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t r = u / CT;
        if (r != cur_r) {
            if (cur_r != 0xFFFFFFFFu) {
                cb_pop_front(CB_COEF, N_COEF);
            }
            cur_r = r;
            cb_wait_front(CB_COEF, N_COEF);
        }
        cb_wait_front(CB_Y, 1);
        cb_wait_front(CB_X, N_STREAMS);
        cb_reserve_back(CB_OUT, N_STREAMS);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            tile_regs_acquire();
#if defined(MPF7_DBG) && MPF7_DBG == 5  // timing bisect: no copies, no math (the data-movement + pack floor)
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            pack_tile(D_OUT, CB_OUT);
            tile_regs_release();
            continue;
#endif
            load(CB_Y, 0, D_Y);
            reconfig_data_format_srca(CB_X);
            copy_init(CB_X);
            for (uint32_t i = 0; i < N_STREAMS; ++i) {
                copy_tile(CB_X, i, D_X0 + i);
            }
            copy_tile(CB_COEF, 2 * s, D_K);  // (CB_COEF: the same fp32 format / unpack mode as CB_X)
            copy_tile(CB_COEF, 2 * s + 1, D_L);
#if defined(MPF7_DBG) && MPF7_DBG == 3
            // bisect: out = post_s * y through the LLK's own broadcast multiply on the same full-sync DST slots
            load(CB_Y, 0, D_OUT);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(D_OUT, D_K);
#else
            sfpu_bcast_col_init();  // col-0 lane mask (LREG6) + the broadcast replay; the copies may reprogram addr
                                    // mods
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, mpf7_mix_tile, D_Y, D_K, D_OUT, VectorMode::None)));
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            pack_tile(D_OUT, CB_OUT);
            tile_regs_release();
        }
        cb_push_back(CB_OUT, N_STREAMS);
        cb_pop_front(CB_X, N_STREAMS);
        cb_pop_front(CB_Y, 1);
    }
    cb_pop_front(CB_COEF, N_COEF);
}
