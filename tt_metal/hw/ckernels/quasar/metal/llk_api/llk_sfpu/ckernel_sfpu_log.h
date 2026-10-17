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

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x = setexp(in, 127);  // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Minimax cubic approximation of ln(x) on x in [1, 2), Horner form:
    // x * (x * (x * A + B) + C) + D
    // A: 0.10968964, B: -0.72910421, C: 2.11263230, D: -1.49277612
    // max |error| over [1, 2] = 4.4e-04
    ////////////////////////////
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.11263230f) + -1.49277612f;

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
    sfpi::vFloat vConstLn2 = 0.69314718f;
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
    sfpi::vConstFloatPrgm0 = 0.69314718f;  // ln2

    // Minimax cubic for ln on [1, 2); see _calculate_log_body_.
    sfpi::vConstFloatPrgm1 = 0.10968964f;
    sfpi::vConstFloatPrgm2 = -0.72910421f;
}

// Blackhole metal-layer natural log (calculate_log_body / calculate_log / log_init), carried over for
// the ported kernels that call it (erfinv). Only reset_counters is spelled the Quasar way.
template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en, bool IS_BASE_TWO = false>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat a, const std::uint32_t log_base_scale_factor) {
    if constexpr (!FAST_APPROX) {
        // normalise a (-0.0 and subnormals become +0.0). This must run before
        // the exponent read below so that -0.0 follows the same path as +0.0
        // and log(±0.0) returns -inf, matching torch.
        a = a * 1.0f + 0.0f;
    }

    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);
    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();

    // m in [0.75, 1.5). Compute log1p(m - 1) for m - 1 in [-0.25, 0.5).
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

        // Handle special cases:
        //
        //   input 0.0  -> -inf
        //   input +inf -> +inf
        //   input NaN  -> NaN
        //
        // In the non-fast path, earlier normalisation maps -0.0 and subnormals
        // to +0.0. On Blackhole addexp(a, -1) wraps exponent 0 to 255, so zero
        // becomes +inf; Quasar's SFPDIVP2 clamps the underflowing exponent, so
        // zero stays +0 and gets an explicit branch below. Exponent 255 values
        // (Inf/NaN) are left unchanged on both.
        a = sfpi::addexp(a, -1);

        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
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
            result = e_float * TWO_TO_M23 + r * sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        } else {
            result = e_float * sfpi::vConstFloatPrgm0 + r;

            if constexpr (HAS_BASE_SCALING) {
                result *= sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
            }
        }

        // For +inf, result is positive, so result * +inf gives +inf. NaNs
        // either skip the main block or propagate here.
        // Quasar: !sfpi::is_finite spelled out. SFPI lowers its nearby() compare to a CC-only SFPIADD into
        // read-only LREG8, which leaves the lane mask unchanged on Quasar.
        v_if(sfpi::exexp(a, sfpi::ExponentMode::Biased) >= 255) { result *= a; }
        v_endif;
        // Quasar: zero cannot ride that multiply, because SFPDIVP2 clamped it to +0 (finite)
        // where Blackhole wrapped it to +inf; log(0) = -inf is set explicitly (emulator:
        // erfinv(+-1) returned +-9.29, i.e. erfinv through log(0) ~ -88, instead of +-inf).
        v_if(a == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    v_endif;

    return result;
}

// Whether BF16 DEST runs the generated log10 kernel as one call over the whole tile.
inline constexpr bool log10_bf16_whole_tile = false;
// The stock log10 kernel needs no BF16 setup.
template <bool bf16_kernel>
inline void log10_bf16_tile_init() {}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8,
    bool IS_BASE_TWO = false>
inline void calculate_log(std::uint32_t log_base_scale_factor) {
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
