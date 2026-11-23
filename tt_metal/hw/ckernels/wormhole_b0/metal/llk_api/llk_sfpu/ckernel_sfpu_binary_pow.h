// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
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
 * Parity limit: the odd/even test below converts pow through vSMag16, which saturates at
 * +/-32767, so any larger |pow| reads as non-integer and a negative base returns NaN
 * where the true result is finite: (-1)**40000 gives NaN rather than 1. The inline claim
 * that "large powers will approach 0/Inf" does not hold when |base| == 1.
 *
 * @note This function assumes that the programmable constants are set to the following values:
 * - vConstFloatPrgm0 = 1.4426950408889634f;
 * - vConstFloatPrgm1 = -127.0f;
 *
 * @see Moroz et al. 2022 - "Simple Multiple Precision Algorithms for Exponential Functions"
 *      ( https://doi.org/10.1109/MSP.2022.3157460 )
 */
template <bool is_fp32_dest_acc_en>
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
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat exp_f32 = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

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

    sfpi::vInt zii = sfpi::exexp(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x7f800000 in paper
    sfpi::vInt zif = sfpi::exman(sfpi::as<sfpi::vFloat>(z));  // Note: z & 0x007fffff in paper

    // Compute formula in Horner form
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0xf94ee7) + zif), sfpi::RoundMode::Nearest);
    sfpi::vFloat d3 =
        sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(sfpi::vInt(0x560e) + zif), sfpi::RoundMode::Nearest);

    d2 = d1 * d2;
    zif = _float_to_int32_positive_(d2 * d3);

    // Restore exponent
    zii = sfpi::as<sfpi::vInt>(sfpi::setexp(sfpi::as<sfpi::vFloat>(zif), 127U + zii));

    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(zii);

    // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
    // Check valid base range
    auto pow_int = sfpi::convert<sfpi::vSMag16>(
        pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
    auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

    v_if(base < 0.0f) {  // negative base
        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        // To get the sign bit of result, we can shift last bit of pow_int to the 1st bit
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != pow) {  // negative base and non-integer power => set to NaN
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    // setexp/exexp map 0 to log2 = -127, so 0**p evaluates as 2**(-127p).
    // Must follow the negative-base branch: SFPU `<` is sign-bit based, so
    // v_if(base < 0) classes -0 as negative and would overwrite the 0 with the
    // complex-result NaN; -0 is a zero, not a negative base.
    // Fill 0 for every non-zero exponent, then narrow to the negative ones.
    // v_and tightens the enclosing predicate in place, so it costs one compare
    // where a second flat v_if would also save and restore the lane mask.
    // pow == 0 fails the != 0 gate and keeps 1.
    v_if(absbase == 0.f) {
        v_if(pow != 0.f) {
            y = 0.0f;
            v_and(pow < 0.f);
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest.
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }

    return y;
}

/**
 * @brief base**pow on an fp32 DST.
 *
 * Same two-step log2/exp2 shape as _sfpu_binary_power_21f_, but with a wider log. The
 * zero-base block below is fp32-only: _sfpu_binary_power_21f_ still carries the original
 * one verbatim, so on a bf16 DST pow(0, NaN) returns +0 and pow(0, pow < 0) writes NaN.
 * See #53922, deferred there on tt-llk#675.
 *
 * @param base The base value (sfpi::vFloat vector), can be any floating point number
 * @param pow The exponent/power value (sfpi::vFloat vector), can be any floating point number
 *
 * @return sfpi::vFloat Result of base**pow
 *
 * @note Two consumers, both via _sfpu_binary_power_: calculate_sfpu_binary_pow, and
 * calculate_rpow, which passes its scalar as `base`. So ttnn.rpow(y, 0.0) is pow(0, y)
 * and the zero-base semantics below apply to it too.
 *
 * Special Cases:
 * - base = +/-0, pow > 0: Returns +/-0 for finite pow, preserving the base sign only for odd integer pow; +inf returns
 * +0
 * - base = +/-0, pow < 0: Returns +/-inf for finite pow, preserving the base sign only for odd integer pow; -inf
 * returns +inf
 * - base = +/-0, pow = +/-0: Returns 1
 * - base = +/-0, pow = +/-NaN: Returns NaN
 * - base < 0, pow = integer: Returns proper signed result (negative if odd power)
 * - base < 0, pow = non-integer: Returns NaN (complex result)
 * - Overflow/underflow: Clamped to appropriate limits
 *
 * Parity limit: the same vSMag16 saturation described on _sfpu_binary_power_21f_ above.
 * In this body it also costs the zero-base sign: a -0 base loses it for odd pow in
 * 32767 < |pow| < 2**24, because the saturated pow reads as non-integer and the
 * negative-base branch overwrites y with a positive NaN before the fills below copy its
 * sign. A +0 base is unaffected, and the magnitude stays correct throughout.
 */
sfpi_inline sfpi::vFloat _sfpu_binary_power_f32_(sfpi::vFloat base, sfpi::vFloat pow) {
    // base**pow = 2**(pow*log2|base|) with pow*log2|base| = z_hi + z_lo, z_hi exact (ckernel_sfpu_pow_df.h),
    // so the error no longer grows with |pow*log2 m|.
    sfpi::vFloat z_hi, z_lo;
    _sfpu_pow_z_df_(sfpi::abs(base), pow, z_hi, z_lo);

    sfpi::vFloat y = _sfpu_pow2_df_(z_hi, z_lo);

    // |pow| removes a -0 exponent: convert<vSMag16> would round trip that back to something the
    // bit-exact compare below reports as non-integer (on BH, not on WH) and gives NaN instead of 1
    // for a -0 base. Kept on WH too, where it passes only because that conversion happens to
    // preserve the sign of a -0, which is not a guarantee.
    sfpi::vFloat abs_pow = sfpi::abs(pow);

    v_if(base < 0.0f) {  // negative base
        // Post-processing: ensure that special values (e.g. 0**0, -1**0.5, ...) are handled correctly
        // Check valid base range
        auto pow_int = sfpi::convert<sfpi::vSMag16>(
            abs_pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
        auto pow_rounded = sfpi::convert<sfpi::vFloat>(pow_int, sfpi::RoundMode::Nearest);

        // If pow is odd integer then result is negative
        // If power is even, then result is positive
        y = sfpi::setsgn2(y, pow_int);

        // Check for integer power, if it is not then overwrite result with NaN
        v_if(pow_rounded != abs_pow) {  // negative base and non-integer power => set to NaN
            y = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    // The pow_df reduction maps 0 to log2 = -127 (as setexp/exexp did), so 0**p evaluates as 2**(-127p).
    // Must follow the negative-base branch: SFPU `<` is sign-bit based, so
    // v_if(base < 0) classes -0 as negative and would overwrite the 0 with the
    // complex-result NaN; -0 is a zero, not a negative base.
    // SFPU `== 0` is bit-exact, so matching -0 needs an abs. Recompute |base| here:
    // keeping it live across the pow_df core would not fit in the 8 LRegs.
    // Fill 0 for every non-zero exponent, then narrow to the negative ones, which
    // IEEE defines as +inf. v_and tightens the enclosing predicate in place, so it
    // costs one compare where a second flat v_if would also save and restore the
    // lane mask. Gating on |pow| lets both signed zeros keep the mainline's 1.
    // Both fills take their sign from y rather than writing a positive constant, because
    // IEEE keeps the sign of a zero base through an odd integer exponent.
    sfpi::vFloat abs_end = sfpi::abs(base);
    v_if(abs_end == 0.f) {
        v_if(abs_pow != 0.f) {
            y = sfpi::copysgn(sfpi::vFloat(0.0f), y);
            v_and(pow < 0.f);
            y = sfpi::copysgn(sfpi::vFloat(std::numeric_limits<float>::infinity()), y);
        }
        v_endif;
        v_if(sfpi::is_nan(pow)) { y = std::numeric_limits<float>::quiet_NaN(); }
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
    return _sfpu_binary_power_f32_(base, pow);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const uint in0_offset = dst_index_in0 * dst_tile_size_sfpi;
    const uint in1_offset = dst_index_in1 * dst_tile_size_sfpi;
    const uint out_offset = dst_index_out * dst_tile_size_sfpi;

    // Unrolled for fp32 only. The bf16 body runs slower (measured +17% at unroll 4).
    // On BH, this unroll is perf-neutral.
    if constexpr (is_fp32_dest_acc_en) {
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in0 = sfpi::dst_reg[in0_offset];
            sfpi::vFloat in1 = sfpi::dst_reg[in1_offset];

            sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

            sfpi::dst_reg[out_offset] = result;
            sfpi::dst_reg++;
        }
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in0 = sfpi::dst_reg[in0_offset];
            sfpi::vFloat in1 = sfpi::dst_reg[in1_offset];

            sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

            sfpi::dst_reg[out_offset] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.442695f;
    sfpi::vConstFloatPrgm1 = -127.0f;
}

}  // namespace sfpu
}  // namespace ckernel
