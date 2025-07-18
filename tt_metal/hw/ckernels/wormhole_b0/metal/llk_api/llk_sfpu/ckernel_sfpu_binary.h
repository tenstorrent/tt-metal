// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Helper function for _sfpu_binary_power_
// This function is based _float32_to_int32_, but expects a positive input, which allows us to optimize
// away several lines (and make it faster)
sfpi_inline sfpi::vInt _float_to_int32_positive_(sfpi::vFloat in) {
    sfpi::vInt result;
    sfpi::vInt exp = exexp(in);  // extract exponent
    v_if(exp < 0) { result = 0; }
    v_elseif(exp > 30)  // overflow occurs above this range
    {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<int32_t>::max();
    }
    v_else {
        // extract mantissa
        sfpi::vInt man = exman8(in);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23;  // 23 is number of mantissa in float32
        man = sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), shift));

        result = man;
    }
    v_endif;
    return result;
}

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
 * @note This functions assumes that the programmable constants are set to the following values:
 * - vConstFloatPrgm0 = 1.4426950408889634f;
 * - vConstFloatPrgm1 = -127.0f;
 * - vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow) {
    // THe algorithm works in two steps:
    // 1) Compute log2(base)
    // 2) Compute base**pow = 2**(pow * log2(base))

    // Step 1: Compute log2(base)
    // Normalize base to calculation range
    sfpi::vFloat x = setsgn(base, 0);  // set base as positive
    x = sfpi::setexp(x, 127);          // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vInt exp = sfpi::exexp(base);
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;
    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

    // De-normalize to original range
    const sfpi::vFloat vConst1Ln2 = sfpi::vConstFloatPrgm0;        // 1.4426950408889634f;
    sfpi::vFloat log2_result = expf + series_result * vConst1Ln2;  // exp correction: ln(1+x) + exp*ln(2)

    // Step 2: Compute base**pow = 2**(pow * log2(base))
    // If (base, exponent) => (0, +inf) or (base, exponent) => (N, -inf) then ooutput should be 0
    // However, intermediary values can overflow, which leads to output increasing again instead of
    // staying at 0.
    // This overflows happens when zff < -127. Therefore, we clamp zff to -127.
    sfpi::vFloat zff = pow * log2_result;
    const sfpi::vFloat low_threshold = sfpi::vConstFloatPrgm1;
    v_if(zff < low_threshold) { zff = low_threshold; }
    v_endif;

    zff = addexp(zff, 23);                                                     // * 2**23 (Mn)
    sfpi::vInt z = _float_to_int32_positive_(zff + sfpi::vFloat(0x3f800000));  // (bias + x * log2(a)) * N_m

    sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));
    sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    // Check valid base range
    sfpi::vInt pow_int =
        sfpi::float_to_int16(pow, 0);  // int16 should be plenty, since large powers will approach 0/Inf
    sfpi::vFloat pow_rounded = sfpi::int32_to_float(pow_int, 0);

    // Division by 0 when base is 0 and pow is negative => set to NaN
    v_if((base == 0.f || base == -0.f) && pow < 0.f) {
        y = sfpi::vConstFloatPrgm2;  // negative powers of 0 are NaN, e.g. pow(0, -1.5)
    }
    v_endif

    v_if(base < -0.0f) {  // negative base
        // Check for integer power
        v_if(pow_rounded == pow) {
            // if pow is odd integer, set result to negative
            v_if(pow_int & 0x1) {
                // if negative base and negative pow then x**y = -(abs(x))**(abs(y))
                // `sign` will be used at the end
                y = setsgn(y, 1);
            }
            v_endif;
        }
        v_else {
            // multiplication by NaN gives NaN.
            // Since we are going to multiply the result by `sign` to handle negative bases, we also use
            // `sign` to handle NaN results
            y = sfpi::vConstFloatPrgm2;  // = NaN
        }
        v_endif;
    }
    v_endif;

    return y;
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void calculate_sfpu_binary(const uint dst_offset) {
    if constexpr (BINOP == BinaryOp::POW) {
        for (int d = 0; d < ITERATIONS; d++) {
            constexpr uint dst_tile_size = 32;
            sfpi::vFloat in0 = sfpi::dst_reg[0];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_offset * dst_tile_size];
            sfpi::vFloat result = 0.f;
            result = _sfpu_binary_power_(in0, in1);

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_offset);
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::POW) {
        sfpi::vConstFloatPrgm0 = 1.442695f;
        sfpi::vConstFloatPrgm1 = -127.0f;
        sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
    } else {
        _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
