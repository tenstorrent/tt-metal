// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"

#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE

// Blackhole TTI exp body: exp(x) over ITERATIONS SFPU lanes. Scale is either
// absent or pre-folded into LREG12 by sdpa_calculate_exponential_face_init.
template <int ITERATIONS, bool CLAMP_NEGATIVE, bool is_fp32_dest_acc_en>
void sdpa_calculate_exponential_face() {
    ckernel::sfpu::_sfpu_exp_21f_bf16_tti_</*SCALE_EN*/ false, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>(
        /*unused scale*/ 0);
}

// Loads LREG12 = (SCALE_EN ? scale : 1) * (1/ln2), and LREG13 = c2.
template <bool SCALE_EN, uint32_t scaler_fp32 = 0>
inline void sdpa_calculate_exponential_face_init() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    if constexpr (SCALE_EN) {
        // LREG12 = scaler * (1/log(2)) -- compile-time fold.
        // Workaround bug in exp when loading scale into LREG:
        // To bypass the bug, we pre-load the scale and multiply 1/log(2) by
        // the scale in init.
        constexpr float scale_f = __builtin_bit_cast(float, scaler_fp32);
        constexpr float scaled_inv_ln2 = scale_f * 1.4426950408889634F;
        constexpr uint32_t bits = __builtin_bit_cast(uint32_t, scaled_inv_ln2);
        constexpr uint16_t hi = static_cast<uint16_t>((bits >> 16) & 0xFFFFU);
        constexpr uint16_t lo = static_cast<uint16_t>(bits & 0xFFFFU);

        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo);
    } else {
        // LREG12 = 1/log(2) (0x3FB8AA3B)
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3fb8);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xaa3b);
    }
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);

    // LREG13 = c2 = 4.791750143340323e-15f (0x27aca418)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xa418);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
}

// First-column exp on column 0 only. SFPU does 4 half-face iterations
// (stride-2 access skips column 1), 4× fewer iterations than full-tile.
// The fused scale is carried by LREG12 (= scale*(1/log(2))), preloaded by
// sdpa_exp_tile_init<SCALE_EN=true, scaler_fp32>.
inline void sdpa_calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_6);

    ckernel::sfpu::_sfpu_exp_21f_bf16_tti_<
        /*SCALE_EN*/ false,
        DST_ACCUM_MODE,
        /*CLAMP_NEGATIVE*/ false,
        ITERATIONS_HALF_FACE>(/*unused scale*/ 0);
}

#endif  // ARCH_BLACKHOLE

#ifdef ARCH_WORMHOLE
// Wormhole scaled exp workaround: pre-multiply by the scaler in sfpi inline,
// then call the sfpi accurate exp path directly, avoiding the TTI variant.
namespace _sdpa_detail {
template <int ITERATIONS, bool is_fp32_dest_acc_en, uint32_t DST_STRIDE = 1>
inline void sfpi_exp() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_accurate_<is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        if constexpr (DST_STRIDE == 1) {
            sfpi::dst_reg++;
        } else {
            sfpi::dst_reg += DST_STRIDE;
        }
    }
}

template <int ITERATIONS, bool is_fp32_dest_acc_en, uint16_t scale_bf16, int DST_STRIDE = 1>
inline void mul_then_sfpi_exp() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val = val * sfpi::sFloat16b(static_cast<uint32_t>(scale_bf16));
        sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_accurate_<is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += DST_STRIDE;
    }
}

}  // namespace _sdpa_detail
#endif  // ARCH_WORMHOLE

#endif  // TRISC_MATH

// Full-tile exp (8 SFPU face groups, VectorMode::RC).
// WH uses sfpi accurate exp; BH uses the TTI path configured by sdpa_exp_tile_init.
inline void sdpa_exp_tile(uint32_t idst) {
#ifdef TRISC_MATH
#ifdef ARCH_WORMHOLE
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::sfpi_exp</*ITERATIONS*/ 8, DST_ACCUM_MODE>, idst, VectorMode::RC);
#elif defined(ARCH_BLACKHOLE)
    _llk_math_eltwise_unary_sfpu_params_(
        sdpa_calculate_exponential_face<
            /*ITERATIONS*/ 8,
            /*CLAMP_NEGATIVE*/ false,
            DST_ACCUM_MODE>,
        idst,
        VectorMode::RC);
#endif
#endif  // TRISC_MATH
}

// First-column exp (4 half-face SFPU iterations, VectorMode::C).
// Column vectors only have meaningful data in column 0, so 4× fewer iterations.
inline void sdpa_exp_tile_first_column(uint32_t idst) {
#ifdef TRISC_MATH
#ifdef ARCH_WORMHOLE
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::sfpi_exp</*ITERATIONS*/ 4, DST_ACCUM_MODE, /*DST_STRIDE*/ 2>, idst, VectorMode::C);
#elif defined(ARCH_BLACKHOLE)
    _llk_math_eltwise_unary_sfpu_params_(sdpa_calculate_exponential_first_column, idst, VectorMode::C);
#endif
#endif  // TRISC_MATH
}

// Blackhole exp init: configures LREG12 for unscaled or scaled TTI exp.
template <bool approx, bool SCALE_EN, uint32_t scaler_fp32 = 0>
inline void sdpa_exp_tile_init() {
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::exponential>(
        sdpa_calculate_exponential_face_init<SCALE_EN, scaler_fp32>);
#endif
}

// Backward-compatible unscaled init wrapper.
inline void sdpa_exp_tile_init() {
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ false>();
}

// Arch-dispatched scaled exp: exp(scale * x) over 8 face groups.
//   WH: pre-multiplies by scale via sfpi then calls accurate exp directly.
//   BH: folds scale into LREG12 at init time — one SFPU op fewer per element.
template <uint32_t scaler_fp32>
inline void sdpa_exp_tile_scaled(uint32_t idst) {
#ifdef ARCH_WORMHOLE
    constexpr uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_fp32 >> 16);
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::mul_then_sfpi_exp</*ITERATIONS*/ 8, DST_ACCUM_MODE, scaler_bf16>, idst, VectorMode::RC);
#endif
#elif defined(ARCH_BLACKHOLE)
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ true, scaler_fp32>();
    sdpa_exp_tile(idst);
#endif
}

// Arch-dispatched scaled first-column exp: exp(scale * x) over column 0 only (4 iterations).
template <uint32_t scaler_fp32>
inline void sdpa_exp_tile_first_column_scaled(uint32_t idst) {
#ifdef ARCH_WORMHOLE
    constexpr uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_fp32 >> 16);
#ifdef TRISC_MATH
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr int DST_STRIDE = 2;
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::mul_then_sfpi_exp<ITERATIONS_HALF_FACE, DST_ACCUM_MODE, scaler_bf16, DST_STRIDE>,
        idst,
        VectorMode::C);
#endif
#elif defined(ARCH_BLACKHOLE)
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ true, scaler_fp32>();
    MATH((sdpa_exp_tile_first_column(idst)));
#endif
}
