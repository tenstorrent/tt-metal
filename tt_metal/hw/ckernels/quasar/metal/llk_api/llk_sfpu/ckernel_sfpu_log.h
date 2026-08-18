// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_polyval.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Blackhole-port API surface. Keep the algorithm here so callers can retain their Blackhole bodies; the older
// Quasar _calculate_log_body_ entry points below remain available to existing Quasar kernels.
template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat a, const std::uint32_t log_base_scale_factor) {
    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);

    if constexpr (!FAST_APPROX) {
        a = a * 1.0f + 0.0f;
    }

    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();

    m -= 1.0f;

    v_if(a >= 0.0f) {
        sfpi::vFloat r;
        sfpi::vFloat s = m * m;
        sfpi::vFloat e_float;
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            sfpi::vMag abs_e = sfpi::abs(e);
            r = r * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
            sfpi::vFloat neg_half = -0.5f;
            r = __builtin_rvtt_sfpmad(r.get(), m.get(), neg_half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        } else {
            sfpi::vMag abs_e = sfpi::abs(e);
            sfpi::vFloat neg_quarter = -0.25f;
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
        }

        a = sfpi::addexp(a, -1);

        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        result = e_float * sfpi::vConstFloatPrgm0 + r;

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }

        v_if(sfpi::exexp(a, sfpi::ExponentMode::Biased) - 255 >= 0) { result *= a; }
        v_endif;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    const float LOG_TWO = 0.693147182f;
    const float TWO_TO_M23 = 1.19209290e-7f;
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x = setexp(in, 127);  // set exp to exp bias (put in range of 1-2)

    // XXXXXX ask Namal? if we can derive the coefficients below to higher precision
    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D'):
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    // XXXXX try variants of the below: B'=.7122, C'=2.0869
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(in));

    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::sFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if(in == 0.0F) {  // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base) {
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2 = 0.692871f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if(base == 0.0f) { log_result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    return log_result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_log_() {
    sfpi::vConstFloatPrgm0 = 0.692871f;  // ln2

    // XXXXX could do these to higher precision
    sfpi::vConstFloatPrgm1 = 0.1058f;
    sfpi::vConstFloatPrgm2 = -0.7166f;
}

}  // namespace sfpu
}  // namespace ckernel
