// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

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
 *
 * @return sfpi::vFloat Result of base**pow
 *
 * Special Cases:
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
template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat absbase = setsgn(base, 0);       // set base as positive
    sfpi::vFloat x = sfpi::setexp(absbase, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);

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

    z_f32 = addexp(z_f32, 23);  // equal to multiplying by 2**23
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias);

    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    sfpi::vInt pow_int =
        sfpi::float_to_int16(pow, 0);  // int16 should be plenty, since large powers will approach 0/Inf
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // Division by 0 when base is 0 and pow is negative => set to NaN
    v_if((absbase == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
    }
    v_endif;

    v_if(base < 0.0f) {  // negative base
        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = setsgn(y, pow_int << 31);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        y = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_binary_power_61f_(sfpi::vFloat base, sfpi::vFloat pow) {
    // The algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat abs_base = sfpi::abs(base);        // set base as positive
    sfpi::vFloat x = sfpi::setexp(abs_base, 127);   // set exp to exp bias (put base in range of 1-2)

    // 5th degree polynomial approx - REMEZ algorithm over [1,2]
    sfpi::vFloat series_result =
        x * (x * (x * (x * (x * 0.03101577f - 0.28807408f) + 1.1286426f) - 2.45830873f) + 3.5271965f) - 1.94046315f;

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    sfpi::vFloat exp_f32 = sfpi::int32_to_float(exp, 0);

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

    sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // 61f
    // Normalize mantissa field into a fractional value in [0,1)
    sfpi::vFloat frac = sfpi::int32_to_float(zif, 0) * sfpi::vFloat(1.1920929e-7f);

    // Evaluate degree-6 polynomial coefficients using Horner’s rule
    // Note: Unlike exp_21f, in exp_61f all polynomial coefficients are floating-point values.
    // In exp_21f, the paper mixes integer and float constants to perform bit-level manipulation of the exponent and
    // mantissa fields (using bit manipulation techniques - BMT) for exactness. In exp_61f, all coefficients are
    // floating-point values derived from the Chebyshev polynomial approach, making the implementation simpler and
    // purely mathematical without integer-based operations.
    sfpi::vFloat poly =
        ((((((0.0002170391f * frac) + 0.001243946f) * frac + 0.0096788315f) * frac + 0.055483369f) * frac +
          0.24022982f) *
             frac +
         0.69314699f) *
            frac +
        1.0000000018f;

    // Restore exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(poly, 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    sfpi::vInt pow_int =
        sfpi::float_to_int16(pow, 0);  // int16 should be plenty, since large powers will approach 0/Inf
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // Division by 0 when base is 0 and pow is negative => set to NaN
    v_if((abs_base == 0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
    }
    v_endif;

    v_if(base < 0.0f) {  // negative base
        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn(y, pow_int << 31);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = sfpi::vConstFloatPrgm2;
        }
        v_endif;
    }
    v_endif;

    return y;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

// is_fp32_dest_acc_en == false
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);
}

// is_fp32_dest_acc_en == true
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_61f_(base, pow);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = -127.0f;
    sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
}

}  // namespace sfpu
}  // namespace ckernel
