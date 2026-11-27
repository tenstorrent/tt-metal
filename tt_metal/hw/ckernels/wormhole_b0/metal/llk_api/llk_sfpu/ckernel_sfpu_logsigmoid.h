// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)), in one unary SFPU pass.

#ifdef INP_FLOAT32
inline constexpr bool logsigmoid_input_is_fp32 = true;
#else
inline constexpr bool logsigmoid_input_is_fp32 = false;
#endif

// The fp32 residual is also needed for a bf16 input with an fp32 DEST: the exponential is not rounded
// to bf16 there, so a bf16-tuned residual would show at fp32 resolution.
template <bool is_fp32_dest_acc_en>
inline constexpr bool logsigmoid_wants_fp32_residual = logsigmoid_input_is_fp32 || is_fp32_dest_acc_en;

/**
 * @brief log1p(t) for two data on t in [0, 1], to fp32 precision: t - t^2/2 + t^3 * S(t), S of degree 8.
 *
 * @param y0, y1: The residuals, written by this function
 * @param t0, t1: exp(-|x|) for each of the two data
 */
sfpi_inline void logsigmoid_residual_fp32_x2(sfpi::vFloat& y0, sfpi::vFloat& y1, sfpi::vFloat t0, sfpi::vFloat t1) {
    sfpi::vFloat r0 = 0x1.05283ap-8f;
    sfpi::vFloat r1 = r0;

    sfpi::vFloat c7 = -0x1.6ec08cp-6f;
    r0 = r0 * t0 + c7;
    r1 = r1 * t1 + c7;
    sfpi::vFloat c6 = 0x1.e1a2d8p-5f;
    r0 = r0 * t0 + c6;
    r1 = r1 * t1 + c6;
    sfpi::vFloat c5 = -0x1.9c57a8p-4f;
    r0 = r0 * t0 + c5;
    r1 = r1 * t1 + c5;
    sfpi::vFloat c4 = 0x1.15ac66p-3f;
    r0 = r0 * t0 + c4;
    r1 = r1 * t1 + c4;
    sfpi::vFloat c3 = -0x1.52b252p-3f;
    r0 = r0 * t0 + c3;
    r1 = r1 * t1 + c3;

    r0 = r0 * t0 + sfpi::vConstFloatPrgm2;
    r1 = r1 * t1 + sfpi::vConstFloatPrgm2;
    r0 = r0 * t0 + sfpi::vConstFloatPrgm1;
    r1 = r1 * t1 + sfpi::vConstFloatPrgm1;
    r0 = r0 * t0 + sfpi::vConstFloatPrgm0;
    r1 = r1 * t1 + sfpi::vConstFloatPrgm0;

    sfpi::vFloat t20 = t0 * t0;
    sfpi::vFloat t21 = t1 * t1;
    r0 = t0 * r0 + -0.5f;
    r1 = t1 * r1 + -0.5f;
    y0 = t20 * r0 + t0;
    y1 = t21 * r1 + t1;
}

/**
 * @brief log1p(t) for two data on t in [0, 1], to bfloat16 precision: t * P(t), with P(0) = 1 imposed.
 *
 * @param y0, y1: The residuals, written by this function
 * @param t0, t1: exp(-|x|) for each of the two data
 */
sfpi_inline void logsigmoid_residual_bf16_x2(sfpi::vFloat& y0, sfpi::vFloat& y1, sfpi::vFloat t0, sfpi::vFloat t1) {
    sfpi::vFloat r0 = sfpi::vConstFloatPrgm2;
    sfpi::vFloat r1 = r0;
    r0 = r0 * t0 + sfpi::vConstFloatPrgm1;
    r1 = r1 * t1 + sfpi::vConstFloatPrgm1;
    r0 = r0 * t0 + sfpi::vConstFloatPrgm0;
    r1 = r1 * t1 + sfpi::vConstFloatPrgm0;
    r0 = r0 * t0 + 1.0f;
    r1 = r1 * t1 + 1.0f;
    y0 = t0 * r0;
    y1 = t1 * r1;
}

/**
 * @brief Computes logsigmoid for two data. exp is fed -|x|, so nothing overflows and there is no range split.
 *
 * @tparam is_fp32_dest_acc_en: If true, DEST is fp32 and the result is not rounded to bfloat16
 * @param in: DEST offset of the first datum; the second is at in + 1
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline void logsigmoid_x2(const std::uint32_t in, sfpi::vFloat& result0, sfpi::vFloat& result1) {
    sfpi::vFloat x0 = sfpi::dst_reg[in];
    sfpi::vFloat x1 = sfpi::dst_reg[in + 1];
    sfpi::vFloat n0 = sfpi::setsgn(x0, 1);
    sfpi::vFloat n1 = sfpi::setsgn(x1, 1);
    sfpi::vFloat residual0;
    sfpi::vFloat residual1;
    if constexpr (logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        sfpi::vFloat t0 = _sfpu_exp_fp32_accurate_(n0);
        sfpi::vFloat t1 = _sfpu_exp_fp32_accurate_(n1);
        logsigmoid_residual_fp32_x2(residual0, residual1, t0, t1);
    } else {
        // Keep <is_fp32_dest_acc_en> (false here): the final convert<vFloat16b> rounds away exp21's
        // systematic bias. <true> measured worse (mean error and correctly rounded count).
        sfpi::vFloat t0 = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(n0);
        sfpi::vFloat t1 = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(n1);
        logsigmoid_residual_bf16_x2(residual0, residual1, t0, t1);
    }
    if constexpr (logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        // Re-read x from DEST: holding it across exp and the residual runs the fp32 path out of LREGs.
        x0 = sfpi::dst_reg[in];
        x1 = sfpi::dst_reg[in + 1];
    }
    sfpi::vFloat xm0 = sfpi::min(x0, 0.0f);
    sfpi::vFloat xm1 = sfpi::min(x1, 0.0f);
    result0 = xm0 - residual0;
    result1 = xm1 - residual1;
    if constexpr (!logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        // exp21's clamp loses NaN: restore the input NaN.
        v_if(sfpi::is_nan(x0)) { result0 = x0; }
        v_endif;
        v_if(sfpi::is_nan(x1)) { result1 = x1; }
        v_endif;
    }
    if constexpr (!is_fp32_dest_acc_en) {
        result0 = sfpi::convert<sfpi::vFloat16b>(result0, sfpi::RoundMode::Nearest);
        result1 = sfpi::convert<sfpi::vFloat16b>(result1, sfpi::RoundMode::Nearest);
    }
}

/**
 * @brief Computes logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)) in place.
 *
 * @tparam APPROXIMATION_MODE: Ignored
 * @tparam is_fp32_dest_acc_en: If true, DEST is fp32 and the result is not rounded to bfloat16
 * @tparam ITERATIONS: Number of iterations for given face; must be even
 * @note Requires @ref logsigmoid_init: the residual reads the programmable constants.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
    static_assert(ITERATIONS % 2 == 0);
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS / 2; ++d) {
        sfpi::vFloat result0;
        sfpi::vFloat result1;
        logsigmoid_x2<is_fp32_dest_acc_en>(0, result0, result1);
        sfpi::dst_reg[0] = result0;
        sfpi::dst_reg[1] = result1;
        sfpi::dst_reg += 2;
    }
}

/**
 * @brief Loads the three low residual coefficients into the programmable constants.
 *
 * @tparam APPROXIMATION_MODE: Ignored
 * @tparam is_fp32_dest_acc_en: Must match @ref calculate_logsigmoid; it selects the residual
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void logsigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        // S(0) = 1/3, S'(0) = -1/4, S''(0)/2 = 1/5
        sfpi::vConstFloatPrgm0 = 0x1.555554p-2f;
        sfpi::vConstFloatPrgm1 = -0x1.fffdc4p-3f;
        sfpi::vConstFloatPrgm2 = 0x1.995bb8p-3f;
    } else {
        // c1, c2, c3 of P(t) = 1 + c1 t + c2 t^2 + c3 t^3, log1p(t) ~ t * P(t): fitted, P(0) = 1 imposed
        sfpi::vConstFloatPrgm0 = -0x1.f574b8p-2f;
        sfpi::vConstFloatPrgm1 = 0x1.0afacap-2f;
        sfpi::vConstFloatPrgm2 = -0x1.4057aep-4f;
    }
}

}  // namespace ckernel::sfpu
