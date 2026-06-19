// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

static const float PI = 3.14159274101257324f;
static const float PI_2 = 1.5707963705062866f;
static const float PI_4 = 0.7853981852531433f;
static const float FRAC_1_PI = 0.31830987334251404f;
static const float FRAC_2_PI = 0.6366197466850281f;

template <bool is_fp32_dest_acc_en>
static sfpi::vFloat sfpu_tan(sfpi::vFloat x, sfpi::vInt i);

template <>
sfpi_inline sfpi::vFloat sfpu_tan<true>(sfpi::vFloat a, sfpi::vInt i) {
    sfpi::vFloat s = a * a;

    // tan(x) for x in [-PI/4, PI/4]
    sfpi::vFloat t = 0x1.fa9f82p-9f;
    t = t * s + 0x1.2b404p-10f;
    t = t * s + 0x1.4787dp-7f;
    t = t * s + 0x1.620abcp-6f;
    t = t * s + 0x1.ba5716p-5f;
    t = t * s + 0x1.111072p-3f;
    t = t * s + 0x1.555556p-2f;
    t = t * s;

    sfpi::vFloat r = t * a + a;

    v_if(i < 0) {
        // Compensated residual for the reciprocal-correction branch.
        // This preserves precision when tan(x) is near its poles.
        s = sfpi::vConstNeg1 * r + a;
        sfpi::vFloat negative_x = sfpi::copyman(-1.0f, r);
        s = t * a + s;

        // Approximate reciprocal of -r using quadratic initial estimate.
        const float k0 = 0.3232325017452239990234375f;
        const float k1 = 1.4545459747314453125f;
        const float k2 = 2.121212482452392578125f;

        sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vInt>(r);
        t = k1 + k0 * negative_x;
        sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);
        t = k2 + t * negative_x;
        scale *= 0.5f;

        // Newton-Raphson refinement.
        sfpi::vFloat e = sfpi::vConst1 + negative_x * t;
        t = t * e + t;
        t = t * scale;

        // Reconstruct tan from corrected reciprocal terms.
        r = r * t + sfpi::vConst1;
        r = s * t + r;
        r = r * t + t;
    }
    v_endif;

    return r;
}

template <>
sfpi_inline sfpi::vFloat sfpu_tan<false>(sfpi::vFloat a, sfpi::vInt i) {
    sfpi::vFloat s = a * a;

    // tan(x) for x in [-PI/4, PI/4]
    sfpi::vFloat t = 0x1.4f1f4ep-4f;
    t = t * s + 0x1.02b98p-3f;
    t = t * s + 0x1.55953p-2f;
    t = t * s;

    sfpi::vFloat r = t * a + a;

    v_if(i < 0) {
        // Approximate reciprocal of -r using quadratic initial estimate.
        const float k0 = 0.3232325017452239990234375f;
        const float k1 = 1.4545459747314453125f;
        const float k2 = 2.121212482452392578125f;

        sfpi::vFloat negative_x = sfpi::copyman(-1.0f, r);
        t = k1 + k0 * negative_x;
        sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vInt>(r);
        t = k2 + t * negative_x;
        sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

        // Newton-Raphson refinement.
        sfpi::vFloat e = sfpi::vConst1 + negative_x * t;
        scale *= 0.5f;
        t = t * e + t;
        r = t * scale;
    }
    v_endif;

    return r;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tangent() {
    // Constants for four-stage Cody-Waite reduction with -PI/2 = P0 + P1 + P2 + P3
    const float P0 = -0x1.92p+0f;   // representable as bf16
    const float P1 = -0x1.fbp-12f;  // representable as fp16

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vInt i;

        sfpi::vFloat inv_pio2 = sfpi::vConstFloatPrgm2;

        // j = round(v / (PI/2))
        // j = v * (2/PI) + 1.5*2**23 shifts the mantissa bits to give round-to-nearest.
        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        sfpi::vFloat rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        sfpi::vFloat j =
            __builtin_rvtt_sfpmad(v.get(), inv_pio2.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // We need the LSB of the integer later, to determine the sign of the result.
        i = sfpi::reinterpret<sfpi::vInt>(j);

        // Shift mantissa bits back; j is now round(v / (PI/2)) in fp32.
        j += -rounding_bias;

        i <<= 31;

        // Four-stage Cody-Waite reduction; a = v - j * (PI/2).
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        a = sfpu_tan<is_fp32_dest_acc_en>(a, i);

        if constexpr (!is_fp32_dest_acc_en) {
            a = sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sine() {
    // 1. Reduce argument using a four-stage Cody-Waite reduction to the interval [-PI/2, PI/2].
    // 2. Use odd symmetry (sin(-x) = -sin(x)) via quadrant/sign tracking.
    // 3. Evaluate sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, PI/2].

    // Constants for four-stage Cody-Waite reduction with -PI = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+1f;   // representable as bf16
    const float P1 = -0x1.fbp-11f;  // representable as fp16

    sfpi::vFloat C3, C2, C1, C0;

    // Coefficients are chosen per destination precision target for sin(a) on [0, PI/2].
    if (is_fp32_dest_acc_en) {
        C3 = 0x1.5dc908p-19f;
        C2 = -0x1.9f70fp-13f;
        C1 = 0x1.110edap-7f;
        C0 = -0x1.55554cp-3f;
    } else {
        C2 = -0x1.8b10a4p-13f;
        C1 = 0x1.10c2a2p-7f;
        C0 = -0x1.5554a4p-3f;
    }

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        sfpi::vFloat rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        sfpi::vFloat inv_pi = sfpi::vConstFloatPrgm2;

        // Compute j = round(v / PI).
        // First, j = v * (1 / PI) + 1.5*2^23 shifts the mantissa bits to give round-to-nearest.
        sfpi::vFloat j = __builtin_rvtt_sfpmad(v.get(), inv_pi.get(), rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);

        // At this point, the mantissa bits of j contain the integer.
        // Store for later; the LSB determines the sign of the result.
        sfpi::vInt q = sfpi::reinterpret<sfpi::vInt>(j);
        // Shift mantissa bits back; j is now round(v / PI) in fp32.
        j = j - rounding_bias;

        // Four-stage Cody-Waite reduction; a = v + j * -PI.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        q <<= 31;
        sfpi::vFloat s = a * a;
        a = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) ^ q);

        sfpi::vFloat r;
        if (is_fp32_dest_acc_en) {
            r = C3 * s + C2;
            r = r * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
        } else {
            r = C2 * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosine() {
    // 1. Build an odd quadrant index j for PI/2-based reduction.
    // 2. Reduce to a in [-PI/2, PI/2] and fold sign from the quadrant parity.
    // 3. Evaluate sin(a) polynomial and use identity cos(x) = sin(x + PI/2).

    // Constants for four-stage Cody-Waite reduction with -PI/2 = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+0f;   // representable as bf16
    const float P1 = -0x1.fbp-12f;  // representable as fp16

    sfpi::vFloat C3, C2, C1, C0;

    if constexpr (is_fp32_dest_acc_en) {
        // Constants for sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, PI/2].
        C3 = 0x1.5dc908p-19f;
        C2 = -0x1.9f70fp-13f;
        C1 = 0x1.110edap-7f;
        C0 = -0x1.55554cp-3f;
    } else {
        C2 = -0x1.8b10a4p-13f;
        C1 = 0x1.10c2a2p-7f;
        C0 = -0x1.5554a4p-3f;
    }

    const float ROUNDING_BIAS = 12582912.0f;
    const float NEG_ROUNDING_BIAS = -12582912.0f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Force v * (1/PI) + 0.5 to compile as a single SFPMAD sequence for consistent instruction scheduling.
        sfpi::vFloat half = sfpi::sFloat16b(0.5f);  // 0.5
        sfpi::vFloat inv_pi = sfpi::vConstFloatPrgm2;
        sfpi::vFloat neg_one = sfpi::vConstNeg1;

        // Start from j = v * (1 / PI) + 0.5; after bias-round and 2*j - 1, j is an odd quadrant index.
        // ROUNDING_BIAS shifts mantissa bits to perform round-to-nearest.
        sfpi::vFloat j = __builtin_rvtt_sfpmad(v.get(), inv_pi.get(), half.get(), SFPMAD_MOD1_OFFSET_NONE);

        // sfpi::vFloat rounding_bias;
        // rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        // j = __builtin_rvtt_sfpmad(v.get(), one, rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);

        j = j + ROUNDING_BIAS;

        // At this point, the mantissa bits of j contain the rounded integer.
        // Store for later; the LSB tracks quadrant parity for sign selection.
        sfpi::vInt q = sfpi::reinterpret<sfpi::vInt>(j);

        j = j + NEG_ROUNDING_BIAS;

        sfpi::vFloat two = sfpi::sFloat16b(2.0f);
        j = __builtin_rvtt_sfpmad(j.get(), two.get(), neg_one.get(), SFPMAD_MOD1_OFFSET_NONE);

        // Four-stage Cody-Waite reduction; a = v + j * -PI / 2.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        q <<= 31;
        sfpi::vFloat s = a * a;
        a = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) ^ q);

        sfpi::vFloat r;
        if constexpr (is_fp32_dest_acc_en) {
            r = C3 * s + C2;
            r = r * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
        } else {
            r = C2 * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_atan(sfpi::vFloat val) {
    sfpi::vFloat t0 = sfpi::abs(val);
    sfpi::vFloat result = sfpi::vConst0;

    // If input is NaN then output must be NaN as well
    sfpi::vInt exponent = sfpi::exexp(val, sfpi::ExponentMode::NoDebias);
    sfpi::vInt mantissa = sfpi::exman(val);
    v_if(exponent == 255 && mantissa != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_else {
        sfpi::vFloat absval_minus_1 = t0 - sfpi::vConst1;

        v_if(absval_minus_1 > 0.0f) { t0 = sfpu_reciprocal<false>(t0); }
        v_endif;

        sfpi::vFloat t1 = t0 * t0;

        if constexpr (!is_fp32_dest_acc_en) {
            // Low-degree minimax polynomial (Sollya) for reduced-precision destination path.
            // > fpminimax(atan(x), [|1,3,5,7|], [|single...|], [2^(-40); 1], relative);
            t1 = PolynomialEvaluator::eval(
                t1,
                0.999787867069244384765625f,
                -0.325808584690093994140625f,
                0.1555790007114410400390625f,
                -4.4326744973659515380859375e-2f);
        } else {
            // Higher-degree minimax polynomial (Sollya) for fp32 destination path.
            // > fpminimax(atan(x), [|1,3,5,7,9,11,13,15,17|], [|single...|], [2^(-40); 1], relative);
            t1 = PolynomialEvaluator::eval(
                t1,
                sfpi::vConst1,
                -0.3333314359188079833984375f,
                0.19993579387664794921875f,
                -0.14209578931331634521484375f,
                0.1066047251224517822265625f,
                -7.5408883392810821533203125e-2f,
                4.3082617223262786865234375e-2f,
                -1.62907354533672332763671875e-2f,
                2.90188402868807315826416015625e-3f);
        }

        t1 = t1 * t0;

        v_if(absval_minus_1 > 0.0f) { t1 = PI_2 - t1; }
        v_endif;

        result = sfpi::copysgn(t1, val);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpu_atan<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in);

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_asin_ratio_poly_direct(sfpi::vFloat val) {
    // Polynomial in Horner form for asin(z)/z in u=z^2, evaluated over reduced intervals.
    // asin(z) = z * P(u).
    sfpi::vFloat z2 = val * val;
    sfpi::vFloat ratio;
    if constexpr (!is_fp32_dest_acc_en) {
        // Low-degree polynomial for reduced-precision destination path; |z| <= 5/8 => u=z^2 <= (5/8)^2.
        // Single-precision fit to asin(sqrt(u))/sqrt(u) (same Horner depth as atan low path). Regenerate with:
        // > fpminimax(asin(sqrt(x))/sqrt(x), [|0,1,2,3|], [|single...|], [2^(-40); (5/8)^2], relative);
        ratio = PolynomialEvaluator::eval(
            z2,
            0.999978601932525634765625f,
            0.16771225631237030029296875f,
            0.06381262838840484619140625f,
            0.083148844540119171142578125f);
    } else {
        // Higher-degree series coefficients for fp32 destination path.
        ratio = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,
            0.16666666666666666f,
            0.075f,
            0.044642857142857144f,
            0.030381944444444444f,
            0.022372159090909091f,
            0.017352764423076923f,
            0.01396484375f,
            0.011551800896139705f,
            0.009761609529194078f);
    }
    return val * ratio;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_asin_range_reduced(sfpi::vFloat val) {
    // Use symmetry + range transform for better accuracy near |x| ~= 1:
    // asin(x) = sign(x) * [pi/2 - 2*asin(sqrt((1-|x|)/2))].
    sfpi::vFloat abs_v = sfpi::abs(val);
    sfpi::vFloat asin_abs = PI_2;

    v_if(abs_v <= 0.625f) { asin_abs = sfpu_asin_ratio_poly_direct<APPROXIMATION_MODE, is_fp32_dest_acc_en>(abs_v); }
    v_else {
        sfpi::vFloat t = (1.0f - abs_v) * 0.5f;
        sfpi::vFloat root = sfpu_sqrt_custom<APPROXIMATION_MODE>(t);
        sfpi::vFloat asin_root = sfpu_asin_ratio_poly_direct<APPROXIMATION_MODE, is_fp32_dest_acc_en>(root);
        asin_abs -= 2.0f * asin_root;
    }
    v_endif;

    return sfpi::copysgn(asin_abs, val);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool IS_ACOS, int ITERATIONS = 8>
inline void calculate_asin_acos_impl() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
        v_if(sfpi::abs(v) <= sfpi::vConst1) {
            sfpi::vFloat a = sfpu_asin_range_reduced<APPROXIMATION_MODE, is_fp32_dest_acc_en>(v);
            if constexpr (IS_ACOS) {
                result = PI_2 - a;
            } else {
                result = a;
            }
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_asin() {
    calculate_asin_acos_impl<APPROXIMATION_MODE, is_fp32_dest_acc_en, false, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_acos() {
    calculate_asin_acos_impl<APPROXIMATION_MODE, is_fp32_dest_acc_en, true, ITERATIONS>();
}

// Magic seed locally tuned for this sequence, targeting 0 < x < 2^24.
// fp32 path: exhaustively validated maxulperr < 0.94 for normal fp32 2^-126 <= x <= 2^103.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_gt0_(sfpi::vFloat x) {
    constexpr uint MAGIC_SEED = 0xfef392e0;

    // initial estimate y = -reciprocal(x)
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(MAGIC_SEED - sfpi::reinterpret<sfpi::vInt>(x));
    sfpi::vFloat e = x * y + 1.0f;

    if constexpr (is_fp32_dest_acc_en) {
        y = y * e + y;
        e = x * y + 1.0f;
    }
    sfpi::vFloat p = e * e + e;
    y = -y;
    y = y * p + y;

    return y;
}

// computes exp(abs(x))/4 without overflow
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_quarter_exp_abs_(sfpi::vFloat x) {
    // j = x * log2(e); i = round(abs(j)); j = (float)i;
    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    sfpi::vFloat r, f, c1;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 0.038877178f;
        r = r * f + 0.168174848f;
        i += 125;
        r = r * f + sfpi::vConstFloatPrgm2;
        c1 = sfpi::reinterpret<sfpi::vFloat>(
            sfpi::reinterpret<sfpi::vInt>(sfpi::vConst1) - 613);  // 0x3f7ffd9b = 0.999963462f
        r = r * f + c1;

    } else {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)_hi
        f = j * -1.42860677e-6f + f;         // f = f - j * ln(2)_lo

        r = 1.37805939e-3f;
        r = r * f + 8.37312452e-3f;
        r = r * f + 4.16695364e-2f;
        r = r * f + 1.66664720e-1f;
        r = r * f + sfpi::vConstFloatPrgm2;
        i += 125;
        r = r * f + 1.0f;
    }

    // Handle a * log2(e) >= 130, while propagating NaN.
    sfpi::vFloat y = a * std::numeric_limits<float>::infinity();
    r = r * f + 1.0f;

    v_if(i < 255) {
        // Keep reconstruction quarter-scaled: scale is 0.25 * 2**i. Avoids
        // materialising 2**i directly near overflow boundary.
        y = r * sfpi::reinterpret<sfpi::vFloat>(i << 23);
    }
    v_endif;

    return y;
}

// t = exp(a); cosh(a) = 0.5 * (t + 1/t)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::setsgn(x, 0);
        sfpi::vFloat q = _sfpu_quarter_exp_abs_<is_fp32_dest_acc_en>(a);
        sfpi::vFloat r = _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(q);
        sfpi::vFloat y = q + q;
        r *= 0.125f;
        sfpi::vInt q_exp = sfpi::exexp(q);
        v_if(q_exp < 24) { y += r; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

// computes expm1(abs(x))/4 without overflow
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_quarter_expm1_abs_(sfpi::vFloat x) {
    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;  // j = x * log2(e)
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    sfpi::vFloat r, s, f, w, y, scale, bias, c0;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 8.361816406e-03f;
        r = r * f + 4.177856445e-02f;
        s = f * f; // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        c0 = 0.5f;
        r = __builtin_rvtt_sfpmad(r.get(), f.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    } else {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)_hi
        f = j * -1.42860677e-6f + f;         // f = f - j * ln(2)_lo

        r = 1.974105835e-04f;
        r = r * f + 1.393107930e-3f;
        r = r * f + 8.333439939e-3f;
        r = r * f + 4.166680202e-2f;
        s = f * f; // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 4.999999702e-1f;
    }

    w = 0.25f;
    r = r * s + f;

    // Keep reconstruction quarter-scaled: scale is 0.25 * 2**i. Avoids
    // materialising 2**i directly near overflow boundary.
    scale = sfpi::reinterpret<sfpi::vFloat>((i << 23) + sfpi::reinterpret<sfpi::vInt>(w));
    bias = scale - w;
    // Handle a * log2(e) >= 130, while propagating NaN.
    y = a * std::numeric_limits<float>::infinity();

    v_if(i < 130) { y = r * scale + bias; }
    v_endif;

    return y;
}

// a = abs(x); t = expm1(a); sinh(a) = 0.5 * (t + t / (t + 1))
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_sinh_(sfpi::vFloat x) {
    sfpi::vFloat q = _sfpu_quarter_expm1_abs_<is_fp32_dest_acc_en>(x);
    sfpi::vFloat e = 4.0f * q + 1.0f;

    sfpi::vFloat r = _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(e);

    // t < 2^-25: t + 1 rounds to 1, so sinh(x) rounds to x. Since q = t / 4, this is q < 2^-27.
    sfpi::vFloat y = x;
    sfpi::vInt q_exp = sfpi::exexp(q);
    v_if(q_exp >= -27) {
        // t >= 2^25: t + 1 rounds to t, so sinh(abs(x)) = expm1(abs(x)) / 2 = 2q.
        y = q + q;
        v_if(q_exp < 23) {
            // Middle range: sinh(abs(x)) = 0.5t + 0.5t/(t+1), with t = 4q.
            y = y * r + y;
        }
        v_endif;
    }
    v_endif;
    return sfpi::copysgn(y, x);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat y = _sfpu_sinh_<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);

        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void sine_init() {
    // P2 and P3 of four-part Cody-Waite reduction by PI.
    sfpi::vConstFloatPrgm0 = -0x1.51p-21f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-33f;

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;
}

template <bool APPROXIMATION_MODE>
void cosine_init() {
    // P2 and P3 of four-part Cody-Waite reduction by PI/2.
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;
}

template <bool APPROXIMATION_MODE>
void tangent_init() {
    // P2 and P3 of four-part Cody-Waite reduction by PI/2.
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;

    sfpi::vConstFloatPrgm2 = FRAC_2_PI;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void cosh_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;    // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 4.99999851e-1f;   // c2
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 0.500122011f;          // c2
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void sinh_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;    // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 1.666667163e-1f;  // c1
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 1.666259766e-01f;      // c1
    }
}

template <bool APPROXIMATION_MODE>
void atan_init() {
    // Initialisation for use of sfpu_reciprocal<false>.
    sfpu_reciprocal_init<false>();
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_sine_maclaurin_series_(sfpi::vFloat val) {
    // Good for [-pi:pi]
    // Maclaurin series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    sfpi::vFloat tmp = val;
    // x
    sfpi::vFloat output = tmp;
    // x^3/3!
    tmp = tmp * val * val;
    output += -0.166666666 * tmp;
    // x^5/5!
    tmp = tmp * val * val;
    output += 0.0083333333 * tmp;
    // x^7/7!
    tmp = tmp * val * val;
    output += -0.0001984126 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^9/9!
        tmp = tmp * val * val;
        output += 0.0000027557 * tmp;
        // x^11/11!
        tmp = tmp * val * val;
        output += -0.00000002505 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_cosine_maclaurin_series_(sfpi::vFloat val) {
    // Good for [-pi:pi]
    // Maclaurin series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    sfpi::vFloat output = 1.0f;
    // x^2/2!
    sfpi::vFloat tmp = val * val;
    output += -0.5 * tmp;
    // x^4/4!
    tmp = tmp * val * val;
    output += 0.0416666666 * tmp;
    // x^6/6!
    tmp = tmp * val * val;
    output += -0.0013888888 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^8/8!
        tmp = tmp * val * val;
        output += 0.0000248015 * tmp;
        // x^10/10!
        tmp = tmp * val * val;
        output += -0.0000002755 * tmp;
    }

    // Write out output
    return output;
}

// Legacy implementation.
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sine_(const int iterations) {
    // SFPU microcode
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        auto whole_v = sfpi::convert<sfpi::vSMag16>(v, sfpi::RoundMode::Nearest);
        auto whole_v_float = sfpi::convert<sfpi::vFloat>(whole_v, sfpi::RoundMode::Nearest);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);
        v_if((whole_v & 1) != 0) {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Legacy implementation.
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_cosine_(const int iterations) {
    // SFPU microcode
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        auto whole_v = sfpi::convert<sfpi::vSMag16>(v, sfpi::RoundMode::Nearest);
        auto whole_v_float = sfpi::convert<sfpi::vFloat>(whole_v, sfpi::RoundMode::Nearest);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);
        v_if((whole_v & 1) != 0) {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Definitions_in_terms_of_logarithms
// acosh(x) = log(x + sqrt(x^2 - 1))
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_acosh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        v_if(inp < sfpi::vConst1) { sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(inp == sfpi::vConst1) { sfpi::dst_reg[0] = sfpi::vConst0; }
        v_else {
            sfpi::vFloat tmp = inp * inp;
            tmp = tmp - sfpi::vConst1;
            tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
            tmp = tmp + inp;
            sfpi::dst_reg[0] = _calculate_log_body_no_init_(tmp);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// asinh(x) = log(x + sqrt(x^2 + 1))
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_asinh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat tmp = inp * inp + sfpi::vConst1;
        tmp = _calculate_sqrt_body_<APPROXIMATION_MODE>(tmp);
        tmp = tmp + sfpi::abs(inp);
        auto res = _calculate_log_body_no_init_(tmp);
        v_if(inp < sfpi::vConst0) { res = -res; }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atanh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat abs_inp = sfpi::abs(inp);
        sfpi::vFloat res;
        v_if(abs_inp > sfpi::vConst1) { res = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(abs_inp == sfpi::vConst1) {
            sfpi::vFloat inf = std::numeric_limits<float>::infinity();
            res = sfpi::copysgn(inf, inp);
        }
        v_else {
            sfpi::vFloat num = sfpi::vConst1 + inp;
            sfpi::vFloat den = sfpi::vConst1 - inp;
            sfpi::vFloat tmp = sfpu_reciprocal_iter<APPROXIMATION_MODE ? 0 : 2>(den);
            tmp = sfpi::copysgn(tmp, den);
            if constexpr (is_fp32_dest_acc_en || APPROXIMATION_MODE) {
                den = tmp;
            } else {
                den = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
            }
            num = num * den;
            den = _calculate_log_body_no_init_(num);
            res = 0.5f * den;
        }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_inverse_hyperbolic() {
    sqrt_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void init_atanh() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
