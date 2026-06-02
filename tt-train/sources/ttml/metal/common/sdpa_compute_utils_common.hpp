// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"

#ifdef TRISC_MATH

// Unscaled TTI exp body: exp(x) over ITERATIONS SFPU lanes (no scale fold).
// SCALE_EN is always false here; scale is either absent (BW path) or pre-folded
// into LREG12 at init time (BH FW path) or applied via sfpi mul before this (WH FW path).
template <int ITERATIONS, bool CLAMP_NEGATIVE, bool is_fp32_dest_acc_en>
void sdpa_calculate_exponential_face() {
    ckernel::sfpu::_sfpu_exp_21f_bf16_tti_</*SCALE_EN*/ false, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>(
        /*unused scale*/ 0);
}

// Loads LREG12 = 1/ln2 and LREG13 = c2 for the 21f bf16 polynomial.
// Identical to the SCALE_EN=false branch of calculate_exponential_full_face_init (FW)
// and to calculate_exponential_full_face_init (BW). Used by both sdpa_exp_tile_init
// (shared) and sdpa_exp_tile_init<SCALE_EN=false> (FW).
inline void sdpa_calculate_exponential_face_init() {
#ifdef ARCH_BLACKHOLE
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
#endif  // ARCH_BLACKHOLE

    // LREG12 = 1/log(2) (0x3FB8AA3B)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3fb8);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xaa3b);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);

    // LREG13 = c2 = 4.791750143340323e-15f (0x27aca418)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xa418);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
}

#endif  // TRISC_MATH

// Full-tile unscaled exp (8 SFPU face groups, VectorMode::RC).
// Replaces exp_full_tile in both sdpa_compute_utils.hpp and sdpa_bw_compute_utils.hpp.
inline void sdpa_exp_tile(uint32_t idst) {
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(
        sdpa_calculate_exponential_face<
            /*ITERATIONS*/ 8,
            /*CLAMP_NEGATIVE*/ false,
            DST_ACCUM_MODE>,
        idst,
        VectorMode::RC);
#endif  // TRISC_MATH
}

// Unscaled exp init: configures the SFPU polynomial with LREG12 = 1/ln2.
// Replaces exp_full_tile_init() in sdpa_bw_compute_utils.hpp and the
// SCALE_EN=false instantiation of exp_full_tile_init<> in sdpa_compute_utils.hpp.
// For the scaled BH path, use sdpa_exp_tile_init<SCALE_EN=true> from sdpa_compute_utils.hpp.
inline void sdpa_exp_tile_init() {
#ifdef TRISC_MATH
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::exponential>(sdpa_calculate_exponential_face_init);
#endif
}
