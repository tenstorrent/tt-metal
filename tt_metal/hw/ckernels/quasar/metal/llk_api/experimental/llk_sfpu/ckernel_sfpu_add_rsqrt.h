// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel::sfpu {

/**
 * @brief Reciprocal square root of x: the reciprocal arms of Blackhole's _calculate_sqrt_body_.
 *
 * Algorithm SQRT_23-bits, or SQRT_10-bits when APPROXIMATE, with the modifications for the reciprocal (Kokosiński,
 * Z., Gepner, P., Moroz, L. et al., "Fast and accurate approximation algorithms for computing floating point square
 * root", Numerical Algorithms (2024), https://doi.org/10.1007/s11075-024-01932-7). Same arithmetic and edge
 * handling as Blackhole: x = +/-0 gives +/-inf, x = +inf gives 0 and a negative x gives NaN. Quasar's own rsqrt has
 * no SQRT_10-bits body (it uses the SFPNONLINEAR LUT when approximate) and sends -0 to NaN, so add_rsqrt carries
 * this one to keep Blackhole's results in both modes.
 *
 * @tparam APPROXIMATE: Use the one-step SQRT_10-bits body instead of SQRT_23-bits, values = <true/false>
 * @tparam FAST_APPROX: Skip the negative-input clamp, leaving the result for x < 0 unspecified, values = <true/false>
 * @param x: The argument.
 * @note Reads vConstIntPrgm0 / vConstFloatPrgm1 (/ vConstFloatPrgm2): run @ref init_add_rsqrt with the same
 *       APPROXIMATE first.
 */
template <bool APPROXIMATE, bool FAST_APPROX>
sfpi_inline sfpi::vFloat _add_rsqrt_body_(const sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);
    sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
    sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
    sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;

    if constexpr (APPROXIMATE) {
        // Algorithm SQRT_10-bits, with modifications for reciprocal.
        sfpi::vFloat c = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat t = sfpi::vConstFloatPrgm1 + negative_y * c;
        // If x != inf and x has a non-zero magnitude (the shift drops the sign, so -0.0 tests as +0.0).
        v_if(infinity_minus_x_bits != 0 && (x_bits << 1) != 0) {
            y = y * t;
            if constexpr (!FAST_APPROX) {
                // This region already excludes +/-0 and +inf, so a bare sign test is enough.
                v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
                v_endif;
            }
        }
        // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
        v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
        v_endif;
    } else {
        // Algorithm SQRT_23-bits, with modifications for reciprocal.
        sfpi::vFloat xy = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat c = negative_y * xy;
        y = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
        xy = x * y;
        negative_y = -y;
        sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);
        sfpi::vFloat half_y = sfpi::addexp(y, -1);
        // If x != inf and x has a non-zero magnitude.
        v_if(infinity_minus_x_bits != 0 && (x_bits << 1) != 0) {
            y = one_minus_xyy * half_y + y;
            if constexpr (!FAST_APPROX) {
                v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
                v_endif;
            }
        }
        v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
        v_endif;
    }
    return y;
}

/**
 * @brief Compute rsqrt(x * INPUT_SCALE + addend) in place over a Dest tile, as Blackhole's calculate_add_rsqrt.
 *
 * Used for RMSNorm's rsqrt(variance + epsilon): the add happens in fp32 inside the SFPU, so the sum never rounds
 * to the Dest width. On a 16-bit Dest the fp32 result is rounded to bf16 by an explicit convert, as on Blackhole
 * (Quasar rounds ties to even). typed_bf16_store stores the converted value as a bf16 vector; the default assigns
 * it back to a float vector first.
 *
 * @tparam APPROXIMATION_MODE: SQRT_10-bits instead of SQRT_23-bits, values = <true/false>
 * @tparam ITERATIONS: Number of SFPU iterations (rows of 4 x 8 lanes) per call.
 * @tparam fp32_dest_acc_en: Dest is 32-bit; when false, round the result to bf16 first, values = <true/false>
 * @tparam FAST_APPROX: Skip the negative-argument clamp (the result for a negative argument is unspecified),
 *         values = <true/false>
 * @tparam typed_bf16_store: Store the bf16-converted result as a bf16 vector (16-bit Dest only), values = <true/false>
 * @tparam INPUT_SCALE: fp32 bits of the factor on x; 1.0 (the default) skips the multiply.
 * @param param0: fp32 bits of the addend.
 * @note Run @ref init_add_rsqrt with the same APPROXIMATION_MODE first.
 */
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    bool fp32_dest_acc_en,
    bool FAST_APPROX,
    bool typed_bf16_store = false,
    std::uint32_t INPUT_SCALE = 0x3f800000u>
inline void calculate_add_rsqrt(const std::uint32_t param0) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_plus_addend;
        if constexpr (INPUT_SCALE == 0x3f800000u) {
            x_plus_addend = x + Converter::as_float(param0);
        } else {
            x_plus_addend = x * Converter::as_float(INPUT_SCALE) + Converter::as_float(param0);
        }

        sfpi::vFloat y = _add_rsqrt_body_<APPROXIMATION_MODE, FAST_APPROX>(x_plus_addend);

        if constexpr (!fp32_dest_acc_en && typed_bf16_store) {
            sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        } else {
            if constexpr (!fp32_dest_acc_en) {
                y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = y;
        }
        sfpi::dst_reg++;
    }
}

/**
 * @brief Program the seed and refinement constants @ref calculate_add_rsqrt reads (Blackhole's sqrt_init).
 *
 * @tparam APPROXIMATION_MODE: Program the SQRT_10-bits constants instead of the SQRT_23-bits ones,
 *         values = <true/false>
 * @note Writes vConstIntPrgm0, vConstFloatPrgm1 and (SQRT_23-bits) vConstFloatPrgm2.
 */
template <bool APPROXIMATION_MODE>
inline void init_add_rsqrt() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstIntPrgm0 = 0x5f0b3892;
        sfpi::vConstFloatPrgm1 = 1.89099014875f;
    } else {
        sfpi::vConstIntPrgm0 = 0x5f1110a0;
        sfpi::vConstFloatPrgm1 = 2.2825186f;
        sfpi::vConstFloatPrgm2 = 2.2533049f;
    }
}

}  // namespace ckernel::sfpu
