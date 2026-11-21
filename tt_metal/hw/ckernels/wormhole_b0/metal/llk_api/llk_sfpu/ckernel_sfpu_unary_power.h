// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_pow_df.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

/**
 * @brief Computes base raised to the power of pow (base**pow)
 *
 * This function implements binary exponentiation using a polynomial approximation algorithm
 * based on "Simple Multiple Precision Algorithms for Exponential Functions [Tips & Tricks]"
 * by Moroz et al. 2022 (https://doi.org/10.1109/MSP.2022.3157460).
 * More specifically, it is the implementation of the `exp_21f` algorithm described in Section 5
 *
 * @param base The base value (sfpi::vFloat vector), can be any floating point number
 * @param pow The exponent/power value (sfpi::vFloat vector), can be any floating point number
 * @tparam IS_POSITIVE_EXPONENT If true, assumes exponent >= 0 (zero base maps to 0 rather than NaN)
 *
 * @return sfpi::vFloat Result of base**pow
 *
 * Special Cases:
 * - base = 0, pow > 0: Returns 0
 * - base = 0, pow < 0: Returns NaN (undefined)
 * - base < 0, pow = integer: Returns proper signed result (negative if odd power)
 * - base < 0, pow = non-integer: Returns NaN (complex result)
 * - Overflow/underflow: Clamped to appropriate limits
 *
 * @note This function assumes that the programmable constants are set to the following values:
 * - vConstFloatPrgm0 = 1.4426950408889634f;
 * - vConstFloatPrgm1 = -127.0f;
 * - vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool IS_POSITIVE_EXPONENT>
sfpi_inline sfpi::vFloat _sfpu_unary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat abs_base = sfpi::abs(base);       // set base as positive
    sfpi::vFloat x = sfpi::setexp(abs_base, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    vFloat series_result = PolynomialEvaluator::eval(x, -0x1.952992p+0f, 0x2.4f5388p+0f, -0xd.e712ap-4f, 0x2.44734p-4f);

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(exp), sfpi::RoundMode::Nearest);

    // De-normalize to original range
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;           // vConst1Ln2 = 1.4426950408889634f;
    sfpi::vFloat log2_result = exp_f32 + series_result * vConst1Ln2;  // exp correction: ln(1+x) + exp*ln(2)

    // Step 2: Compute base**pow = 2**(pow * log2(base))
    // If (base, exponent) => (0, +inf) or (base, exponent) => (N, -inf) then output should be 0
    // However, intermediary values can overflow, which leads to output increasing again instead of
    // staying at 0.
    // This overflow happens when z_f32 < -127. Therefore, we clamp z_f32 to -127.
    sfpi::vFloat z_f32 = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;
    v_if(z_f32 < low_threshold) { z_f32 = low_threshold; }
    v_endif;

    // The paper relies on the following formula (c.f. Sections 1 and 5):
    // z = (bias + x * log2(a)) * N_m; where:
    // N_m = 2**23
    // bias = 0x3f800000

    // In this case, we transform the formula to:
    // z = (bias) * N_m + (x * log2(a)) * N_m
    // where (bias + N_m) = 0x3f800000
    // and (x * log2(a)) * N_m = addexp(z_f32, 23)

    // Notes:
    // - N_m being a power of 2 ensures equivalent results
    // - addexp(z_f32, 23) is used because it translates to a single-cycle SFPDIVP2
    //   instruction with immediate operand (i.e. no extra register used).
    //   (vs. 1 cycle SFPLOADI + 2 cycles MAD)

    z_f32 = sfpi::addexp(z_f32, 23);  // equal to multiplying by 2**23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = sfpi::exexp(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<vSMag>(sfpi::vInt(0xf94ee7) + zif), sfpi::RoundMode::Nearest);
    sfpi::vFloat d3 = sfpi::convert<sfpi::vFloat>(sfpi::as<vSMag>(sfpi::vInt(0x560e) + zif), sfpi::RoundMode::Nearest);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::as<sfpi::vInt>(sfpi::setexp(sfpi::as<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(zii);

    // Negative base handling (for both positive and negative exponents)
    v_if(base < 0.0f) {
        // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
        // Check valid base range
        auto pow_int = sfpi::convert<sfpi::vSMag16>(
            pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
        auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    // setexp/exexp map 0 to log2 = -127, so 0**p evaluates as 2**(-127p).
    // Must follow the sign branch: SFPU `<` is sign-bit based, so v_if(base < 0)
    // fires on -0 and would replace the 0 result with NaN for non-integer p.
    // pow == 0 falls through the guard and keeps 1.
    if constexpr (IS_POSITIVE_EXPONENT) {
        v_if(abs_base == 0.f && pow > 0.f) { y = 0.0f; }
        v_endif;
    } else {
        v_if(abs_base == 0.f) {
            y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
        }
        v_endif;
    }

    // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
    // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
    // rather than 81 (which would have been correct).
    // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
    y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);

    return y;
}

template <bool IS_POSITIVE_EXPONENT>
sfpi_inline sfpi::vFloat _sfpu_unary_power_61f_updated_(const sfpi::vFloat& base, const sfpi::vFloat& pow) {
    sfpi::lreg_pressure _;  // Declare high register pressure

    // Same double-float core as the binary path (ckernel_sfpu_pow_df.h).
    sfpi::vFloat z_hi, z_lo;
    _sfpu_pow_z_df_(sfpi::abs(base), pow, z_hi, z_lo);
    sfpi::vFloat y = _sfpu_pow2_df_<false>(z_hi, z_lo);  // pow is finite, see _sfpu_unary_power_fp32_
    // |base| is recomputed here rather than kept live across the core.
    sfpi::vFloat abs_base = sfpi::abs(base);

    v_if(base < 0.0f) {  // negative base
        // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
        // Check valid base range
        auto pow_int = sfpi::convert<sfpi::vSMag16>(
            pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
        auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    // The pow_df reduction maps 0 to log2 = -127 (as setexp/exexp did), so 0**p evaluates as 2**(-127p).
    // Must follow the sign branch: SFPU `<` is sign-bit based, so v_if(base < 0)
    // fires on -0 and would replace the 0 result with NaN for non-integer p.
    // pow == 0 falls through the guard and keeps 1.
    if constexpr (IS_POSITIVE_EXPONENT) {
        v_if(abs_base == 0.f && pow > 0.f) { y = 0.0f; }
        v_endif;
    } else {
        v_if(abs_base == 0.f) {
            y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
        }
        v_endif;
    }

    return y;
}

template <int ITERATIONS>
inline void _sfpu_unary_power_bf16_(const uint32_t exponent) {
    // Convert exponent to float
    const float pow_scalar = Converter::as_float(exponent);
    const sfpi::vFloat pow = pow_scalar;

    if (pow_scalar >= 0.0f) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_21f_<true>(base, pow);
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_21f_<false>(base, pow);
            sfpi::dst_reg++;
        }
    }
}

template <int ITERATIONS>
inline void _sfpu_unary_power_fp32_(const uint32_t exponent) {
    // Convert exponent to float. An infinite exponent goes in as +-FLT_MAX (its bits minus one): every base gives
    // the same result with both, and the core then needs no per-row check for an infinite pow.
    const bool pow_is_inf = (exponent & 0x7fffffffu) == 0x7f800000u;
    const float pow_scalar = Converter::as_float(pow_is_inf ? exponent - 1 : exponent);
    const sfpi::vFloat pow = pow_scalar;

    if (pow_scalar >= 0.0f) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_61f_updated_<true>(base, pow);
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat base = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_unary_power_61f_updated_<false>(base, pow);
            sfpi::dst_reg++;
        }
    }
}

inline void power_init() {
#if !(defined(MM_THROTTLE) && MM_THROTTLE > 0)
    // calculate_unary_power_iterative's paired store walks dest through physical addr_mod slot 6, advancing
    // by the two rows the pair just wrote (one sfpi row is two dest counter steps), so the row loop needs no
    // separate increment. Unary SFPU sets addr_mod_base, so the store names ADDR_MOD_2, which maps to
    // physical slot 6. A throttled matmul (MM_THROTTLE > 0) programs ADDR_MOD_6 itself, so there the
    // kernel increments dest explicitly and this init leaves the slot alone.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
#endif
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Compute power operation
 *
 * @param exponent The exponent as IEEE 754 float bits (reinterpreted as uint32_t)
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_unary_power(const uint32_t exponent) {
    if constexpr (is_fp32_dest_acc_en) {
        _sfpu_unary_power_fp32_<ITERATIONS>(exponent);
    } else {
        _sfpu_unary_power_bf16_<ITERATIONS>(exponent);
    }
}

/**
 * @brief Raises two adjacent rows to the power (exp_minus_one + 1) by binary exponentiation
 *
 * Each row runs the same multiply sequence as a single-row walk; pairing them shares the walk's
 * branches and lets each row's multiply cover the other's MUL latency, so no NOPs are needed.
 * The last squaring is skipped since its result is never used. Advances dst_reg by two rows: the second
 * store does it through ADDR_MOD_6 as programmed by @ref power_init (an explicit increment under MM_THROTTLE).
 *
 * @param exp_minus_one Exponent minus one (the first factor is the base itself)
 */
sfpi_inline void _sfpu_unary_power_iterative_two_rows_(const uint32_t exp_minus_one) {
    sfpi::vFloat in0 = sfpi::dst_reg[0];
    sfpi::vFloat in1 = sfpi::dst_reg[1];
    sfpi::vFloat result0 = in0;
    sfpi::vFloat result1 = in1;
    for (uint32_t exp = exp_minus_one; exp > 0;) {
        if (exp & 1) {
            result0 *= in0;
            result1 *= in1;
        }
        exp >>= 1;
        if (exp == 0) {
            break;
        }
        in0 *= in0;
        in1 *= in1;
    }
    sfpi::dst_reg[0] = result0;
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
    sfpi::dst_reg[1] = result1;
    sfpi::dst_reg += 2;
#else
    sfpi::dst_reg[1].mode(ADDR_MOD_2) = result1;
#endif
}

/**
 * @brief Compute power operation using iterative approach
 *
 * @tparam APPROXIMATION_MODE Unused; the result is exact repeated multiplication in both modes.
 * @tparam ITERATIONS Number of Dest rows to process; must be even, since rows are processed in pairs.
 * @param exponent Non-negative integer exponent value
 * @note Call @ref power_init after any other SFPU init that writes ADDR_MOD_6; the row pairs advance Dest
 *       through it.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_power_iterative(const uint32_t exponent) {
    static_assert(ITERATIONS % 2 == 0, "rows are processed in pairs");
    // x^0 == 1 for every input
    if (exponent == 0) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::dst_reg[0] = 1.0f;
            sfpi::dst_reg++;
        }
        return;
    }
    // Iterative approach for positive integer exponents
    const uint32_t exp_minus_one = exponent - 1;
    if (__builtin_constant_p(exponent) || exponent == 1) {
        // A constant exponent folds the walk to straight-line multiplies, and x^1 has no walk at all, so
        // unrolling lets the compiler replay-roll the rows.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d += 2) {
            _sfpu_unary_power_iterative_two_rows_(exp_minus_one);
        }
    } else {
        // Runtime exponent: keep the row loop rolled so the bit walk is emitted once, not per row.
#pragma GCC unroll 1
        for (int d = 0; d < ITERATIONS; d += 2) {
            _sfpu_unary_power_iterative_two_rows_(exp_minus_one);
        }
    }
}

inline void sfpu_unary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;
    sfpi::vConstFloatPrgm1 = -127.0f;
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
}

}  // namespace sfpu
}  // namespace ckernel
