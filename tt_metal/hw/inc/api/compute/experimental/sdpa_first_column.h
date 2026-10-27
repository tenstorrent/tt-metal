// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_recip.h"
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

// SFPU helpers for SDPA statistics tiles, where only the first column of each tile holds data.

namespace ckernel {

#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_INIT_FN(reciprocal, sfpu::sfpu_reciprocal_init, (false));
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_recip_first_column, (is_fp32_dest_acc_en), idst, VectorMode::C);
}
#endif

#if defined(TRISC_MATH) || defined(TRISC_PACK)
template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void exp_tile_first_column(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, is_fp32_dest_acc_en>()));
    PACK((exp_packthread_tile_init<SDPA_EXP_APPROX_MODE>()));
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential_first_column,
        (SDPA_EXP_APPROX_MODE, scale_bf16, is_fp32_dest_acc_en),
        idst,
        VectorMode::C);
}
#endif

#ifdef TRISC_MATH
template <bool approx_mode, VectorMode vector_mode = VectorMode::C, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    constexpr bool reuse_cur_max_tile = is_fp32_dest_acc_en && DST_SYNC_MODE == DstSync::SyncHalf;
    exp_tile_init<approx_mode>();
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_fused_max_sub_exp_add_tile,
        (is_fp32_dest_acc_en, reuse_cur_max_tile),
        idst,
        vector_mode,
        scale_bf16);
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void softplus_tile_first_column(uint32_t idst, uint beta, uint beta_reciprocal, uint threshold) {
    softplus_tile_init();
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_softplus_first_column,
        (is_fp32_dest_acc_en),
        idst,
        VectorMode::C,
        beta,
        beta_reciprocal,
        threshold);
}
#endif

#if defined(ARCH_WORMHOLE) && (defined(TRISC_MATH) || defined(TRISC_PACK))
// Inlines the VectorMode::C traversal instead of the generic unary SFPU launcher, whose larger replay program
// would overwrite the no-mop matmul region.
ALWI void recip_tile_first_column_wh_idst0_direct() {
    SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (APPROX, DST_ACCUM_MODE));
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    math::set_addr_mod_base();

#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        ckernel::sfpu::calculate_recip_first_column<DST_ACCUM_MODE>();
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
#endif

}  // namespace ckernel
