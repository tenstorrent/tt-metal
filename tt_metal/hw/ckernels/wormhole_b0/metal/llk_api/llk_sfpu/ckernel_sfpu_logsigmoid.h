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

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)). The exponential and the residual
// stay in one unary SFPU pass: no second DST tile and no copy/abs/neg/exp tile
// plumbing. Both format paths are selected at compile time.

#ifdef INP_FLOAT32
inline constexpr bool logsigmoid_input_is_fp32 = true;
#else
inline constexpr bool logsigmoid_input_is_fp32 = false;
#endif

/**
 * @brief Selects the fp32 residual.
 *
 * The residual needs relative precision whenever it can reach the output at fp32
 * resolution. That is the case for an fp32 input, and also for a bf16 input with
 * an fp32 DEST: there the exponential is not rounded on the way out, so a
 * residual tuned for eight mantissa bits would be exposed at full fp32 width.
 * Gating on the input dtype alone costs about 4.9e-4 relative, or roughly 7,700
 * fp32 ULP, in that second configuration.
 */
template <bool is_fp32_dest_acc_en>
inline constexpr bool logsigmoid_wants_fp32_residual = logsigmoid_input_is_fp32 || is_fp32_dest_acc_en;

/**
 * @brief Evaluates log1p(t) for two data, to fp32 relative precision, on t in [0, 1].
 *
 * `t` is always exp(-|x|), so it cannot leave [0, 1] and 1 + t cannot leave [1, 2].
 * The exponent of 1 + t is therefore always zero, which makes the argument reduction
 * that @ref calculate_log1p_fp32 performs for an arbitrary argument dead weight here.
 *
 * Pulling the two leading terms out analytically keeps the form accurate in the
 * relative sense down to t -> 0, and confines the rounding of each Horner step to S,
 * where t^3 damps it:
 *
 *     log1p(t) = t - t^2/2 + t^3 * S(t),   S of degree 8 on [0, 1]
 *
 * Degree 8 is the knee: degree 6 gives 11 ULP and degree 7 reaches 2 ULP on a thinner
 * margin, while degree 8 matches the full log1p to three digits of relative error.
 *
 * Two data are evaluated in lockstep, following @ref _sfpu_tanh_polynomial_x2_. Six of
 * the nine coefficients are not encodable as immediates and cost a pair of SFPLOADI
 * each; lockstep loads them once for both chains and fills the MAD latency slots. The
 * three low coefficients live in the programmable registers @ref logsigmoid_init loads.
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
 * @brief Evaluates log1p(t) for two data, to bfloat16 precision, on t in [0, 1].
 *
 * Eight mantissa bits do not pay for the fp32 form. `P(0) = 1` is imposed rather than
 * fitted, so that the positive tail — where the residual is the whole answer and is
 * tiny — inherits the relative error of the exponential instead of picking up a fixed
 * polynomial bias.
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
 * @brief Computes logsigmoid for two data.
 *
 * The identity is exact on both halves of the domain: for x >= 0 it reads
 * -log1p(e^-x) = log(1/(1+e^-x)), and for x < 0 it reads x - log1p(e^x) =
 * log(e^x/(1+e^x)). The exponential is fed -|x|, built in one SFPSETSGN, so no
 * intermediate can overflow and there is no range split.
 *
 * @tparam is_fp32_dest_acc_en: If true, DEST is fp32 and the result is not rounded to bfloat16
 * @note For |x| above about 87.3, exp(-|x|) falls below the smallest normal float and
 *       flushes. A positive input there returns +0 rather than a subnormal; torch returns
 *       a negative subnormal up to about 103.97 and +0 beyond. A negative input returns x,
 *       which is the correctly rounded answer.
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline void logsigmoid_x2(sfpi::vFloat x0, sfpi::vFloat x1, sfpi::vFloat& result0, sfpi::vFloat& result1) {
    sfpi::vFloat n0 = sfpi::setsgn(x0, 1);
    sfpi::vFloat n1 = sfpi::setsgn(x1, 1);
    sfpi::vFloat residual0;
    sfpi::vFloat residual1;
    if constexpr (logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        sfpi::vFloat t0 = _sfpu_exp_fp32_accurate_(n0);
        sfpi::vFloat t1 = _sfpu_exp_fp32_accurate_(n1);
        logsigmoid_residual_fp32_x2(residual0, residual1, t0, t1);
    } else {
        sfpi::vFloat t0 = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(n0);
        sfpi::vFloat t1 = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(n1);
        logsigmoid_residual_bf16_x2(residual0, residual1, t0, t1);
    }
    sfpi::vFloat xm0 = sfpi::min(x0, 0.0f);
    sfpi::vFloat xm1 = sfpi::min(x1, 0.0f);
    result0 = xm0 - residual0;
    result1 = xm1 - residual1;
    if constexpr (!logsigmoid_wants_fp32_residual<is_fp32_dest_acc_en>) {
        // exp21 clamps NaN during its integer conversion. Restore the input NaN
        // after the finite fast path so logsigmoid keeps its public semantics.
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
 * @note Call @ref logsigmoid_init first: the residual reads the programmable constants
 *       that init loads, and produces wrong results without it.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
    static_assert(ITERATIONS % 2 == 0);
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS / 2; ++d) {
        sfpi::vFloat result0;
        sfpi::vFloat result1;
        logsigmoid_x2<is_fp32_dest_acc_en>(sfpi::dst_reg[0], sfpi::dst_reg[1], result0, result1);
        sfpi::dst_reg[0] = result0;
        sfpi::dst_reg[1] = result1;
        sfpi::dst_reg += 2;
    }
}

/**
 * @brief Adapter for the legacy binary LLK harness.
 *
 * Its normal placement has in0 == out. Operand B held exp(-x) when the caller computed
 * it; the exponential is computed here now, so it is intentionally ignored.
 *
 * @tparam ITERATIONS: Number of iterations for given face; must be even
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logsigmoid_binary(
    const std::uint32_t dst_index_in0, const std::uint32_t /*dst_index_in1*/, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    static_assert(ITERATIONS % 2 == 0);
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS / 2; ++d) {
        sfpi::vFloat result0;
        sfpi::vFloat result1;
        logsigmoid_x2<is_fp32_dest_acc_en>(
            sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi],
            sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi + 1],
            result0,
            result1);
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result0;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi + 1] = result1;
        sfpi::dst_reg += 2;
    }
}

/**
 * @brief Loads the three low residual coefficients into the programmable registers.
 *
 * @tparam APPROXIMATION_MODE: Ignored
 * @tparam is_fp32_dest_acc_en: Must match the value passed to @ref calculate_logsigmoid,
 *         since it selects which residual, and therefore which coefficients, are used
 * @note Must run before @ref calculate_logsigmoid.
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
        sfpi::vConstFloatPrgm0 = -0x1.f574b8p-2f;
        sfpi::vConstFloatPrgm1 = 0x1.0afacap-2f;
        sfpi::vConstFloatPrgm2 = -0x1.4057aep-4f;
    }
}

}  // namespace ckernel::sfpu
