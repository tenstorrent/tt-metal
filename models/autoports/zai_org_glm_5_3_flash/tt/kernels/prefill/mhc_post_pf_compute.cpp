// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 prefill mHC post-mix — compute (mhc_post_pf_common.hpp): per unit and output stream s,
// out = y * bcast_col(post_s) + sum_i x_i * bcast_col(C[i][s]) with exact fp32 SFPU products / sums (fp32 DEST,
// fp32 unpack-to-dest of the residual and coefficient CBs). Compile-time: CT. Runtime: first unit, unit count.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

#include "mhc_post_pf_common.hpp"

using namespace mpp;

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
        cb_wait_front(CB_X, N_STREAMS);
        cb_reserve_back(CB_OUT, N_OUT);
#ifdef MPP_PRE
        tile_regs_acquire();  // h = sum_i x_i * bcast_col(pre_i)
        load(CB_X, 0, 0);
        load(CB_COEF, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);
        for (uint32_t i = 1; i < N_STREAMS; ++i) {
            load(CB_X, i, 1);
            load(CB_COEF, i, 3);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(1, 3);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_tile(0, CB_OUT);
        tile_regs_release();
#else
        cb_wait_front(CB_Y, 1);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            tile_regs_acquire();
            load(CB_Y, 0, 0);
            load(CB_COEF, s, 3);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 3);
            for (uint32_t i = 0; i < N_STREAMS; ++i) {
                load(CB_X, i, 1);
                load(CB_COEF, N_STREAMS + i * N_STREAMS + s, 3);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(1, 3);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            pack_tile(0, CB_OUT);
            tile_regs_release();
        }
        cb_pop_front(CB_Y, 1);
#endif
        cb_push_back(CB_OUT, N_OUT);
        cb_pop_front(CB_X, N_STREAMS);
    }
    cb_pop_front(CB_COEF, N_COEF);
}
