// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base) {
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2 = 0.69314718f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if(base == 0.0f) { log_result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    return log_result;
}

// Natural log ported from the Blackhole metal layer (calculate_log_body / calculate_log / log_init),
// used by the log op and erfinv. Only reset_counters is spelled the Quasar way.
template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en, bool IS_BASE_TWO = false>
sfpi_inline sfpi::vFloat calculate_log_body(
    sfpi::vFloat a, [[maybe_unused]] const std::uint32_t log_base_scale_factor) {
    if constexpr (!FAST_APPROX) {
        // normalise a (-0.0 and subnormals become +0.0). This must run before
        // the exponent read below so that -0.0 follows the same path as +0.0
        // and log(±0.0) returns -inf, matching torch.
        a = a * 1.0f + 0.0f;
    }

    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);
    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    // e is k << 23 in two's complement: one cast to sign-magnitude, one to float (exact).
    sfpi::vFloat e_float = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(e), sfpi::RoundMode::Nearest);
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();  // a < 0 (and -NaN)

    // m in [0.75, 1.5). Compute log1p(m - 1) for m - 1 in [-0.25, 0.5).
    m -= 1.0f;

    v_if(a >= 0.0f) {
        // +0 -> -inf, +inf -> +inf, +NaN -> NaN in one expression: a - 1/a, where the hardware
        // reciprocal is exact at both ends (approx_recip(+0) = +inf, approx_recip(+inf) = +0).
        if constexpr (FAST_APPROX) {
            // Not normalised: -0.0 reaches here with its sign, and 1/|a| keeps log(-0.0) = -inf.
            result = a - sfpi::approx_recip(sfpi::abs(a));
        } else {
            result = a - sfpi::approx_recip(a);
        }

        // The finite positive lanes overwrite that with the polynomial.
        v_and(a > 0.0f);
        v_and(a < std::numeric_limits<float>::infinity());

        sfpi::vFloat r;
        sfpi::vFloat s = m * m;
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            r = r * m + sfpi::vConstFloatPrgm1;
            r = r * m + sfpi::vConstFloatPrgm2;
            sfpi::vFloat neg_half = -0.5f;
            r = __builtin_rvtt_sfpmad(r.get(), m.get(), neg_half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        } else {
            sfpi::vFloat neg_quarter = -0.25f;
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            r = r * m + sfpi::vConstFloatPrgm2;
        }

        r = r * s + m;
        if constexpr (IS_BASE_TWO) {
            // log2 takes an exact path.  Scaling the finished natural-log sum, as
            // result *= 1/ln(2), also scales the exponent contribution by
            // ln(2) * (1/ln(2)), which does not round to exactly 1 in float, so log2 of
            // an exact power of two came back an ULP low for 46 of the 254 representable
            // exponents.
            //
            // Applying the base change to the mantissa term only leaves the exponent
            // term as e_float * 2^-23.  The exponent arrives here as k << 23 and 2^-23 is
            // a power of two, so that product is exact and log2(2^k) == k for every k.
            // Instruction count is unchanged: one multiply plus one multiply-add, where
            // before it was one multiply-add plus one multiply.
            //
            // This is deliberately not applied to other bases.  For log10 the equivalent
            // constant is log10(2) * 2^-23, which is not exactly representable, and
            // folding it in measured worse than the existing path.
            constexpr float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
            result = e_float * TWO_TO_M23 + r * sfpi::vConstFloatPrgm3;
        } else {
            result = e_float * sfpi::vConstFloatPrgm0 + r;

            if constexpr (HAS_BASE_SCALING) {
                result *= sfpi::vConstFloatPrgm3;
            }
        }
    }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8,
    bool IS_BASE_TWO = false>
inline void calculate_log(std::uint32_t log_base_scale_factor) {
    if constexpr (HAS_BASE_SCALING) {
        // Prgm3 holds the base scale (1 / ln(base)) that calculate_log_body multiplies by.
        sfpi::vConstFloatPrgm3 = sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en, IS_BASE_TWO>(
            sfpi::dst_reg[0], log_base_scale_factor);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
    // e represents k << 23 rather than k, so pre-fold the 2^(-23) factor into
    // the constant used for the final exponent contribution.
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        // Stored separately because the tuned fp32 m^3 and m^4 coefficients are
        // no longer the shared exact 1/3 and -1/4 values used in the bf16 path.
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        // Horner coefficients used by bf16 polynomial
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
