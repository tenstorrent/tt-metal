// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"

#ifdef TRISC_MATH

// Unscaled TTI exp body: exp(x) over ITERATIONS SFPU lanes (no scale fold).
// SCALE_EN is always false; scale is either absent (BW path) or pre-folded
// into LREG12 at init time (BH FW path) or applied via sfpi mul before this (WH FW path).
template <int ITERATIONS, bool CLAMP_NEGATIVE, bool is_fp32_dest_acc_en>
void sdpa_calculate_exponential_face() {
    ckernel::sfpu::_sfpu_exp_21f_bf16_tti_</*SCALE_EN*/ false, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>(
        /*unused scale*/ 0);
}

// Loads LREG12 = 1/ln2 and LREG13 = c2. Used by sdpa_exp_tile_init (unscaled path)
// and by sdpa_exp_tile_init<SCALE_EN=false> (FW unscaled path).
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

// First-column exp on column 0 only. SFPU does 4 half-face iterations
// (stride-2 access skips column 1), 4× fewer iterations than full-tile.
// The fused scale is carried by LREG12 (= scale*(1/log(2))), preloaded by
// sdpa_exp_tile_init<SCALE_EN=true, scaler_fp32>.
inline void sdpa_calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;

#ifdef ARCH_BLACKHOLE
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_6);
#endif  // ARCH_BLACKHOLE

    ckernel::sfpu::_sfpu_exp_21f_bf16_tti_<
        /*SCALE_EN*/ false,
        DST_ACCUM_MODE,
        /*CLAMP_NEGATIVE*/ false,
        ITERATIONS_HALF_FACE>(/*unused scale*/ 0);
}

// Scaled exp init: folds scaler_fp32*(1/log(2)) into LREG12 at compile time.
// Lets the inner SFPU loop skip the per-iteration SFPMULI fix-up — saves one
// cycle per iteration. Used on ARCH_BLACKHOLE only; WH uses the sfpi
// mul+exp workaround path in _sdpa_detail instead.
template <uint32_t scaler_fp32>
inline void sdpa_calculate_exponential_face_init_scaled() {
#ifdef ARCH_BLACKHOLE
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
#endif  // ARCH_BLACKHOLE

    // LREG12 = scaler * (1/log(2)) — compile-time fold.
    constexpr float scale_f = __builtin_bit_cast(float, scaler_fp32);
    constexpr float scaled_inv_ln2 = scale_f * 1.4426950408889634F;
    constexpr uint32_t bits = __builtin_bit_cast(uint32_t, scaled_inv_ln2);
    constexpr uint16_t hi = static_cast<uint16_t>((bits >> 16) & 0xFFFFU);
    constexpr uint16_t lo = static_cast<uint16_t>(bits & 0xFFFFU);

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, hi);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lo);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);

    // LREG13 = c2 = 4.791750143340323e-15f (0x27aca418)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xa418);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
}

// Arch-dispatched sfpi-based scaled exp helpers (WH workaround).
//   WH: the 21f bf16 TTI exp polynomial misbehaves when reached multiple times
//       per kernel invocation (deterministic NaN at the last processed sequence
//       row; root cause not pinned). Workaround: pre-multiply by the scaler in
//       sfpi inline, then call the sfpi 21f exp polynomial directly (avoiding the
//       TTI variant entirely).
//   BH: uses the LREG12 = scaler*(1/log2) fold instead — see sdpa_exp_tile_scaled.
namespace _sdpa_detail {
template <int ITERATIONS, bool is_fp32_dest_acc_en>
inline void mul_then_sfpi_exp(uint16_t scale_bf16) {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val = val * sfpi::sFloat16b(static_cast<uint32_t>(scale_bf16));
        sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_accurate_<is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <uint16_t scale_bf16>
inline void mul_then_sfpi_exp_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
#ifdef ARCH_BLACKHOLE
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_6);
#endif
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val = val * sfpi::sFloat16b(static_cast<uint32_t>(scale_bf16));
        sfpi::vFloat result = ckernel::sfpu::_sfpu_exp_accurate_<DST_ACCUM_MODE>(val);
        if constexpr (!DST_ACCUM_MODE) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += 2;
    }
}
}  // namespace _sdpa_detail

#endif  // TRISC_MATH

// Full-tile unscaled exp (8 SFPU face groups, VectorMode::RC).
// Used by SDPA backward (BW path has no fused scale) and by BH forward
// after sdpa_exp_tile_init<SCALE_EN=true> pre-folds the scale into LREG12.
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

// First-column exp (4 half-face SFPU iterations, VectorMode::C).
// Column vectors only have meaningful data in column 0, so 4× fewer iterations.
inline void sdpa_exp_tile_first_column(uint32_t idst) {
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(sdpa_calculate_exponential_first_column, idst, VectorMode::C);
#endif  // TRISC_MATH
}

// Unscaled exp init: configures SFPU with LREG12 = 1/ln2.
// Used by SDPA backward. For the FW scaled path, use sdpa_exp_tile_init<SCALE_EN=true>.
inline void sdpa_exp_tile_init() {
#ifdef TRISC_MATH
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::exponential>(sdpa_calculate_exponential_face_init);
#endif
}

// Dispatches to the appropriate init based on SCALE_EN.
//   SCALE_EN=false → sdpa_calculate_exponential_face_init (LREG12 = 1/log(2)).
//   SCALE_EN=true  → sdpa_calculate_exponential_face_init_scaled<scaler_fp32>
//                    (LREG12 = scaler*(1/log(2))); `scaler_fp32` ignored when SCALE_EN=false.
template <bool approx, bool SCALE_EN, uint32_t scaler_fp32 = 0>
inline void sdpa_exp_tile_init() {
#ifdef TRISC_MATH
    if constexpr (SCALE_EN) {
        ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::exponential>(
            sdpa_calculate_exponential_face_init_scaled<scaler_fp32>);
    } else {
        ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::exponential>(sdpa_calculate_exponential_face_init);
    }
#endif
}

// Arch-dispatched scaled exp: exp(scale * x) over 8 face groups.
//   WH: pre-multiplies by scale via sfpi then calls unscaled polynomial (TTI workaround).
//   BH: folds scale into LREG12 at init time — one SFPU op fewer per element.
template <uint32_t scaler_fp32>
inline void sdpa_exp_tile_scaled(uint32_t idst) {
#ifdef ARCH_WORMHOLE
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ false>();
    constexpr uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_fp32 >> 16);
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::mul_then_sfpi_exp</*ITERATIONS*/ 8, DST_ACCUM_MODE>, idst, VectorMode::RC, scaler_bf16);
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
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ false>();
    constexpr uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_fp32 >> 16);
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(
        _sdpa_detail::mul_then_sfpi_exp_first_column<scaler_bf16>, idst, VectorMode::C);
#endif
#elif defined(ARCH_BLACKHOLE)
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ true, scaler_fp32>();
    MATH((sdpa_exp_tile_first_column(idst)));
#endif
}
