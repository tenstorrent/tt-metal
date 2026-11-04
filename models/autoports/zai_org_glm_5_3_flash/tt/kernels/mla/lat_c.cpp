// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel latent core — compute: latent = RMSNorm_512(kv_a) (kernels/mla/rmsnorm.hpp, exact fp32 on the SFPU)
// -> KX bf16 tiles in CB_OUT for the cache writer (kernels/mla/lat_dm1.cpp).
// The row scale comes from all KX input tiles; the output is the slice [OOFF, OOFF + ON) (the q-norm cores of the
// one-program half split RMSNorm_1536(q_a)'s 48 output tiles; the latent core writes all of its 16).
// ON = 0: inv only (the one-program half's q-norm core: 1 / rms(q_a) for the folded q_b / I3 phases' row scale).
// Compile-time args: CB_X, CB_NW, CB_INV, CB_OUT, KX, INV_N_BITS, EPS_BITS, ON.
// Runtime args: OOFF.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/rmsnorm.hpp"

void kernel_main() {
    constexpr uint32_t CB_X = get_compile_time_arg_val(0);
    constexpr uint32_t CB_NW = get_compile_time_arg_val(1);
    constexpr uint32_t CB_INV = get_compile_time_arg_val(2);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(3);
    constexpr uint32_t KX = get_compile_time_arg_val(4);
    constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t ON = get_compile_time_arg_val(7);
    const uint32_t OOFF = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_OUT);
    cb_wait_front(CB_X, KX);
    mlan::norm_inv<KX>(CB_X, CB_INV, INV_N_BITS, EPS_BITS);
    cb_wait_front(CB_INV, 1);
    if constexpr (ON == 0) {  // inv only: the fp32 row-scale tile 1 / rms (column 0) -> CB_OUT (fp32)
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_INV);
        copy_tile_to_dst_init_short(CB_INV);
        copy_tile(CB_INV, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_X, KX);
        cb_pop_front(CB_INV, 1);
        return;
    }
    cb_wait_front(CB_NW, ON);
    cb_reserve_back(CB_OUT, ON);
    for (uint32_t k = 0; k < ON; ++k) {
        tile_regs_acquire();
        mlan::norm_tile(CB_X, OOFF + k, CB_INV, CB_NW, k);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        pack_tile(0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, ON);
    cb_pop_front(CB_X, KX);
    cb_pop_front(CB_NW, ON);
    cb_pop_front(CB_INV, 1);
}
