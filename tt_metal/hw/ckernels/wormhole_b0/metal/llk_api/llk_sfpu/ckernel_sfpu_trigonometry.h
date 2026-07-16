// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

// Magic seed locally tuned for this sequence, targeting 0 < x < 2^24.
// fp32 path: exhaustively validated maxulperr < 0.94 for normal fp32 2^-126 <= x <= 2^103.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_gt0_(sfpi::vFloat x) {
    constexpr std::uint32_t MAGIC_SEED = 0xfef392e0;

    // initial estimate y = -reciprocal(x)
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(MAGIC_SEED - sfpi::as<sfpi::vInt>(x));
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

sfpi_inline sfpi::vFloat _sfpu_sqrt_endpoint_(sfpi::vFloat half_d) {
    // SQRT_23-bits from ckernel_sfpu_sqrt.h, specialized for the known
    // non-negative input half_d.  The generic negative and infinity handling
    // is unnecessary here, and the zero path naturally evaluates to zero.
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(half_d) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(0x5f1110a0 - i);

    sfpi::vFloat half_d_y = half_d * y;
    sfpi::vFloat c = (-y) * half_d_y;
    y = y * (2.2825186f + c * (2.2533049f + c));

    half_d_y = half_d * y;
    sfpi::vFloat one_minus_half_d_yy = 1.0f + (-y) * half_d_y;
    return one_minus_half_d_yy * (0.5f * half_d_y) + half_d_y;
}

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
        s = -1.0f * r + a;
        sfpi::vFloat negative_x = sfpi::copyman(-1.0f, r);
        s = t * a + s;

        // Approximate reciprocal of -r using quadratic initial estimate.
        const float k0 = 0.3232325017452239990234375f;
        const float k1 = 1.4545459747314453125f;
        const float k2 = 2.121212482452392578125f;

        sfpi::vInt scale_bits = ~sfpi::as<sfpi::vInt>(r);
        t = k1 + k0 * negative_x;
        sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);
        t = k2 + t * negative_x;
        scale *= 0.5f;

        // Newton-Raphson refinement.
        sfpi::vFloat e = 1.0f + negative_x * t;
        t = t * e + t;
        t = t * scale;

        // Reconstruct tan from corrected reciprocal terms.
        r = r * t + 1.0f;
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
        sfpi::vInt scale_bits = ~sfpi::as<sfpi::vInt>(r);
        t = k2 + t * negative_x;
        sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);

        // Newton-Raphson refinement.
        sfpi::vFloat e = 1.0f + negative_x * t;
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
        i = sfpi::as<sfpi::vInt>(j);

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
    if constexpr (is_fp32_dest_acc_en) {
        C3 = 0x1.5dc908p-19f;
        C2 = -0x1.9f70fp-13f;
        C1 = 0x1.110edap-7f;
        C0 = -0x1.55554cp-3f;
    } else {
        C2 = -0x1.8b10a4p-13f;
        C1 = 0x1.10c2a2p-7f;
        C0 = -0x1.5554a4p-3f;
    }

#pragma GCC unroll 8
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
        sfpi::vInt q = sfpi::as<sfpi::vInt>(j);
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
        a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) ^ q);

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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Force v * (1/PI) + 0.5 to compile as a single SFPMAD sequence for consistent instruction scheduling.
        sfpi::vFloat half = sfpi::sFloat16b(0.5f);  // 0.5
        sfpi::vFloat inv_pi = sfpi::vConstFloatPrgm2;
        sfpi::vFloat neg_one = -1.0f;

        // Start from j = v * (1 / PI) + 0.5; after bias-round and 2*j - 1, j is an odd quadrant index.
        // ROUNDING_BIAS shifts mantissa bits to perform round-to-nearest.
        sfpi::vFloat j = __builtin_rvtt_sfpmad(v.get(), inv_pi.get(), half.get(), SFPMAD_MOD1_OFFSET_NONE);

        // sfpi::vFloat rounding_bias;
        // rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        // j = __builtin_rvtt_sfpmad(v.get(), one, rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);

        j = j + ROUNDING_BIAS;

        // At this point, the mantissa bits of j contain the rounded integer.
        // Store for later; the LSB tracks quadrant parity for sign selection.
        sfpi::vInt q = sfpi::as<sfpi::vInt>(j);

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
        a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) ^ q);

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

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_atan_bf16(sfpi::vFloat val) {
    sfpi::vFloat t0 = sfpi::abs(val);
    sfpi::vFloat result = 0.0f;

    // If input is NaN then output must be NaN as well
    sfpi::vInt exponent = sfpi::exexp(val, sfpi::ExponentMode::Biased);
    sfpi::vInt mantissa = sfpi::exman(val);
    v_if(exponent == 255 && mantissa != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_else {
        sfpi::vFloat absval_minus_1 = t0 - 1.0f;

        v_if(absval_minus_1 > 0.0f) { t0 = sfpu_reciprocal<false>(t0); }
        v_endif;

        sfpi::vFloat t1 = t0 * t0;

        // Low-degree minimax polynomial (Sollya) for reduced-precision destination path.
        // > fpminimax(atan(x), [|1,3,5,7|], [|single...|], [2^(-40); 1], relative);
        t1 = PolynomialEvaluator::eval(
            t1,
            0.999787867069244384765625f,
            -0.325808584690093994140625f,
            0.1555790007114410400390625f,
            -4.4326744973659515380859375e-2f);

        t1 = t1 * t0;

        v_if(absval_minus_1 > 0.0f) { t1 = PI_2 - t1; }
        v_endif;

        result = sfpi::copysgn(t1, val);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_atan_fp32(sfpi::vFloat x) {
    sfpi::vFloat r;
    sfpi::vFloat q;
    sfpi::vFloat s;
    sfpi::vFloat a;
    sfpi::vFloat x_abs;

    x_abs = sfpi::setsgn(x, 0);
    a = x_abs;
    sfpi::vFloat x_abs_m1 = x_abs - 1.0f;

    v_if(x_abs_m1 >= 0.0f) { a = _sfpu_reciprocal_gt0_<true>(a); }
    v_endif;

    // Next we compute the minimax approximation for atan(a).
    {
        q = 0x1.01cp-8f;
        s = a * a;
        sfpi::vFloat c6 = -0x1.4bcp-6f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c6.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c5 = 0x1.93p-5f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c5.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c4 = -0x1.48cp-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c4.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c3 = 0x1.bd4p-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c3.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c2 = -0x1.24p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c2.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c1 = 0x1.99938ap-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.555558p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    sfpi::vFloat half_pi = 0x1.921fb6p+0f;
    sfpi::vFloat t = q * s;
    r = t * a + a;

    // Special cases:
    v_if(x_abs_m1 >= 0.0f) { r = half_pi - r; }
    v_endif;

    r = sfpi::copysgn(r, x);

    return r;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_atan_fp32<APPROXIMATION_MODE>(in);
        } else {
            result = sfpu_atan_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_ratio_poly_direct_bf16(sfpi::vFloat val) {
    // Polynomial in Horner form for asin(z)/z in u=z^2, evaluated over reduced intervals.
    // asin(z) = z * P(u).
    sfpi::vFloat z2 = val * val;
    // Low-degree polynomial for reduced-precision destination path; |z| <= 5/8 => u=z^2 <= (5/8)^2.
    // Single-precision fit to asin(sqrt(u))/sqrt(u) (same Horner depth as atan low path). Regenerate with:
    // > fpminimax(asin(sqrt(x))/sqrt(x), [|0,1,2,3|], [|single...|], [2^(-40); (5/8)^2], relative);
    sfpi::vFloat ratio = PolynomialEvaluator::eval(
        z2,
        0.999978601932525634765625f,
        0.16771225631237030029296875f,
        0.06381262838840484619140625f,
        0.083148844540119171142578125f);
    return val * ratio;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_range_reduced_bf16(sfpi::vFloat val) {
    // Use symmetry + range transform for better accuracy near |x| ~= 1:
    // asin(x) = sign(x) * [pi/2 - 2*asin(sqrt((1-|x|)/2))].
    sfpi::vFloat abs_v = sfpi::abs(val);
    sfpi::vFloat asin_abs = PI_2;

    v_if(abs_v <= 0.625f) { asin_abs = sfpu_asin_ratio_poly_direct_bf16<APPROXIMATION_MODE>(abs_v); }
    v_else {
        sfpi::vFloat t = (1.0f - abs_v) * 0.5f;
        sfpi::vFloat root = sfpu_sqrt_custom<APPROXIMATION_MODE>(t);
        sfpi::vFloat asin_root = sfpu_asin_ratio_poly_direct_bf16<APPROXIMATION_MODE>(root);
        asin_abs -= 2.0f * asin_root;
    }
    v_endif;

    return sfpi::copysgn(asin_abs, val);
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_bf16(sfpi::vFloat val) {
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    v_if(sfpi::abs(val) <= 1.0f) { result = sfpu_asin_range_reduced_bf16<APPROXIMATION_MODE>(val); }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_acos_bf16(sfpi::vFloat val) {
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    v_if(sfpi::abs(val) <= 1.0f) { result = PI_2 - sfpu_asin_range_reduced_bf16<APPROXIMATION_MODE>(val); }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_fp32(sfpi::vFloat x) {
    sfpi::vFloat result;
    sfpi::vFloat x_abs = sfpi::abs(x);
    sfpi::vFloat d = 1.0f - x_abs;
    sfpi::vFloat switchover = 0.5625f;
    sfpi::vFloat half_d = d * 0.5f;
    sfpi::vFloat tmp = x_abs - switchover;

    // asin(x) = pi/2 - 2 * asin(sqrt((1 - |x|) / 2)).
    sfpi::vFloat reduced = _sfpu_sqrt_endpoint_(half_d);

    v_if(tmp < 0.0f) { reduced = x_abs; }
    v_endif;

    // Minimax approximation for asin(reduced) on [0, SWITCHOVER].
    sfpi::vFloat square = reduced * reduced;
    sfpi::vFloat polynomial = 0x1.9e0000p-5f;
    sfpi::vFloat coefficient = 0x1.365f44p-6f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.7dbc50p-5f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.329f7cp-4f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.5556dcp-3f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vFloat neg_two = -2.0f;
    polynomial *= square;
    sfpi::vFloat pio2 = 1.57079637050628662109f;
    result = __builtin_rvtt_sfpmad(polynomial.get(), reduced.get(), reduced.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    v_if(tmp >= 0.0f) { result = pio2 + neg_two * result; }
    v_endif;

    result = sfpi::copysgn(result, x);

    v_if(half_d < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_acos_fp32(sfpi::vFloat x) {
    sfpi::vFloat result;
    sfpi::vFloat x_abs = sfpi::abs(x);
    sfpi::vFloat d = 1.0f - x_abs;
    sfpi::vFloat switchover = 0.5625f;
    sfpi::vFloat half_d = d * 0.5f;
    sfpi::vFloat tmp = x_abs - switchover;

    // acos(x) = 2 * asin(sqrt((1 - x) / 2)).  Reduce to an asin
    // approximation on [-SWITCHOVER, SWITCHOVER].
    sfpi::vFloat reduced = _sfpu_sqrt_endpoint_(half_d);

    v_if(tmp < 0.0f) { reduced = x_abs; }
    v_endif;

    reduced = sfpi::copysgn(reduced, x);
    sfpi::vUInt flip = sfpi::as<sfpi::vUInt>(sfpi::copysgn(sfpi::vFloat(0.0f), tmp));

    // Minimax approximation for asin(reduced).
    sfpi::vFloat square = reduced * reduced;
    sfpi::vFloat polynomial = 0x1.834000p-5f;
    sfpi::vFloat coefficient = 0x1.ca0000p-8f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.15a984p-5f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.6a9354p-5f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.3345f4p-4f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    coefficient = 0x1.555536p-3f;
    polynomial =
        __builtin_rvtt_sfpmad(polynomial.get(), square.get(), coefficient.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    polynomial *= square;
    reduced = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(reduced) ^ flip);
    sfpi::vFloat x_lt_switchover = x - 0.5625f;
    result = __builtin_rvtt_sfpmad(polynomial.get(), reduced.get(), reduced.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    // Map asin(reduced) back to acos(x).
    v_if(x_lt_switchover < 0.0f) { result += 1.57079637050628662109f; }
    v_endif;

    v_if(tmp >= 0.0f) { result += result; }
    v_endif;

    v_if(half_d < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_asin() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_asin_fp32<APPROXIMATION_MODE>(in);
        } else {
            result = sfpu_asin_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_acos() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_acos_fp32<APPROXIMATION_MODE>(in);
        } else {
            result = sfpu_acos_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
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

    sfpi::vFloat r, f;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 0.038877178f;
        r = r * f + 0.168174848f;
        i += 125;
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 0.999963462f;
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
        y = r * sfpi::as<sfpi::vFloat>(i << 23);
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
        s = f * f;  // hide SFPMAD latency
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
        s = f * f;  // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 4.999999702e-1f;
    }

    w = 0.25f;
    r = r * s + f;

    // Keep reconstruction quarter-scaled: scale is 0.25 * 2**i. Avoids
    // materialising 2**i directly near overflow boundary.
    scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));
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
        sfpi::vConstFloatPrgm1 = -0.693145752f;   // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 4.99999851e-1f;  // c2
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
    // Initialisation for use of sfpu_reciprocal<false> by sfpu_atan_bf16.
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

// Self-contained square root for the inverse-hyperbolic kernels.
//
// The shared _calculate_sqrt_body_ stores its magic seed and Newton refinement
// constants in vConstIntPrgm0 / vConstFloatPrgm1 / vConstFloatPrgm2. Those same
// program registers are owned by the log1p polynomial (vConstFloatPrgm0/1/2)
// that asinh/acosh now route through, so the two cannot coexist in one pass.
// This helper bakes the magic seed and refinement constants in as immediates so
// it leaves the program registers untouched for log1p. Input is assumed >= 0
// (always true here: x*x + 1 for asinh, (x-1)*(x+1) for acosh with x >= 1).
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_sqrt_ge0_(sfpi::vFloat x) {
    // Fast inverse-square-root seed (same constant the shared sqrt kernel uses
    // for the high-precision path) followed by Newton-Raphson refinement of
    // y ~= 1 / sqrt(x): y <- y * (1.5 - 0.5 * x * y * y).
    sfpi::vFloat half_x = sfpi::addexp(x, -1);  // 0.5 * x
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(0x5f1110a0 - i);

    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    if constexpr (is_fp32_dest_acc_en) {
        y = y * (1.5f - half_x * y * y);
    }

    // sqrt(x) = x / sqrt(x) = x * (1 / sqrt(x)). One Newton step on the product
    // form (a = x * y; a <- 0.5 * (a + x / a)) is folded as a <- a + 0.5 * (x - a*a) * y.
    sfpi::vFloat a = x * y;
    a = a + 0.5f * (x - a * a) * y;

    // sqrt(0) must be exactly 0; the reciprocal seed produces inf*0 = NaN there.
    v_if(x == 0.0f) { a = 0.0f; }
    v_endif;
    return a;
}

// acosh(x) = log(x + sqrt(x^2 - 1)), reformulated through log1p to remove the
// absorption error at x -> 1+ and the x^2 overflow at large x. Three regions:
//   x < 1            -> NaN
//   x == 1           -> +0
//   1 < x < 1.5      -> log1p((x - 1) + sqrt((x - 1) * (x + 1)))  (small-(x-1) stable)
//   1.5 <= x < 2^28  -> log1p((x + sqrt(x^2 - 1)) - 1)            (safe reconstruction)
//   x >= 2^28        -> ln(2x) = log1p(2x - 1)                    (avoids x^2 overflow)
// LOG1P_LARGE is the threshold past which x^2 - 1 == x^2 to working precision and
// acosh(x) == ln(2x) to <1 ulp; using log1p(2x - 1) also dodges the x^2 overflow
// that makes the classic form return +inf for x >= ~1.84e19.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_acosh() {
    constexpr float LOG1P_LARGE = 268435456.0f;  // 2^28
    constexpr float LN2 = 0.6931471805599453f;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];

        // Build the log1p argument per region, clamping the out-of-domain lanes
        // (x < 1) to a safe value so the shared log1p runs over the whole vector.
        // The argument is materialised to DST before log1p; round-tripping through
        // DST severs the sqrt/reciprocal expression from the log1p polynomial so
        // the SFPU register allocator stays within its reload budget. The x <= 1
        // lanes are overwritten with their exact results afterwards.
        //
        // arg = x - 1 is the common term in every region, so it is computed once
        // and the per-region sqrt term is accumulated onto it:
        //   x >= 2^28        -> arg = x - 1                 (large; acosh ~= LN2 +
        //                                                    log1p(x-1), +LN2 added later)
        //   1 < x < 2^28     -> arg += sqrt((x-1)(x+1))     (sqrt(x^2-1) without the
        //                                                    x^2-1 cancellation)
        // The large region falls through the predicated block and keeps arg = x-1.
        sfpi::vFloat arg = inp - 1.0f;
        v_if(inp < LOG1P_LARGE) { arg = arg + _sfpu_sqrt_ge0_<is_fp32_dest_acc_en>((inp + 1.0f) * arg); }
        v_endif;
        sfpi::dst_reg[0] = arg;

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        // Large region carries the extra ln(2) from acosh(x) ~= LN2 + ln(x).
        v_if(inp >= LOG1P_LARGE) { res = res + LN2; }
        v_endif;

        // Domain fix-ups: x == 1 -> +0, x < 1 -> NaN.
        v_if(inp == 1.0f) { res = 0.0f; }
        v_elseif(inp < 1.0f) { res = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1)), reformulated to remove the
// cancellation at x -> 0 and the x^2 overflow at large |x|. Regions in a = |x|:
//   a < 0.75          -> a * P(a^2), degree-6 minimax polynomial (<=1 ulp)
//   0.75 <= a < 2^28  -> log1p(a + a*a / (1 + sqrt(1 + a*a)))  (cancellation-free)
//   a >= 2^28         -> ln(2a) = LN2 + log1p(a - 1)           (avoids x^2 overflow)
// Sign is restored at the end. The small region is a plain polynomial (no
// sqrt/reciprocal/log1p), which keeps SFPU register pressure within the reload
// budget. The mid region uses a + sqrt(1+a^2) - 1 = a + a^2 / (1 + sqrt(1+a^2))
// so the log1p argument never loses precision through a subtract-1 cancellation;
// the large region exits the x^2 regime entirely so |x| up to fp32 max no longer
// overflows (the old x^2 + 1 produced +inf at ~1.84e19).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_asinh() {
    constexpr float LOG1P_LARGE = 268435456.0f;  // 2^28
    constexpr float LN2 = 0.6931471805599453f;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // Keep only the original input live across the body (matching calculate_acosh,
        // which compiles within the SFPU reload budget). a = |x| and x2 = x*x are
        // recomputed inline rather than held in their own long-lived registers:
        // hoisting them into vFloats pushes the SFPU register allocator past its
        // reload budget and the kernel fails to compile (internal compiler error:
        // maximum number of generated reload insns), so the recompute is deliberate.
        // Build the per-region log1p argument over |x|, clamp |x| < 0.75 lanes to a
        // safe value, materialise to DST, run the shared log1p, then overwrite the
        // |x| < 0.75 lanes with the direct polynomial. Sign is restored from inp.
        sfpi::vFloat inp = sfpi::dst_reg[0];

        // Mid/large region (|x| >= 0.75): asinh(|x|) = log1p(arg). The large
        // sub-region drops the x^2 term (LN2 + log1p(|x| - 1)) to dodge fp32
        // overflow. For the safe sub-region use the cancellation-free identity
        //   |x| + sqrt(1+x^2) - 1 = |x| + x^2 / (1 + sqrt(1+x^2))
        // which avoids the subtract-1 cancellation that otherwise costs ~3-4 ulp
        // near the crossover. Lanes below 0.75 are clamped here and overwritten by
        // the polynomial after log1p.
        sfpi::vFloat arg = 0.0f;
        v_if(sfpi::abs(inp) >= LOG1P_LARGE) { arg = sfpi::abs(inp) - 1.0f; }
        v_elseif(sfpi::abs(inp) >= 0.75f) {
            sfpi::vFloat root = _sfpu_sqrt_ge0_<is_fp32_dest_acc_en>(inp * inp + 1.0f);
            arg = sfpi::abs(inp) + (inp * inp) * _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(1.0f + root);
        }
        v_endif;
        sfpi::dst_reg[0] = arg;

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        v_if(sfpi::abs(inp) >= LOG1P_LARGE) { res = res + LN2; }
        v_endif;

        // Small region (|x| < 0.75): asinh(|x|) = |x| * P(x^2), a degree-6 (in x^2)
        // minimax fit (<=1 ulp on [0, 0.75]). No sqrt/reciprocal/log1p here.
        v_if(sfpi::abs(inp) < 0.75f) {
            sfpi::vFloat s = inp * inp;
            sfpi::vFloat p = 4.375355784e-03f;
            p = p * s + -1.484858524e-02f;
            p = p * s + 2.785361186e-02f;
            p = p * s + -4.417778924e-02f;
            p = p * s + 7.495806366e-02f;
            p = p * s + -1.666652262e-01f;
            p = p * s + 1.000000000e+00f;
            res = sfpi::abs(inp) * p;
        }
        v_endif;

        // res is asinh(|x|) >= 0; restore the original sign.
        res = sfpi::copysgn(res, inp);

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x)), reformulated as
// 0.5 * log1p(2 * x / (1 - x)) to remove the cancellation at x -> 0 and the
// (1 + x)/(1 - x) ratio that loses precision there.
//   |x| > 1   -> NaN
//   |x| == 1  -> copysgn(+inf, x)
//   else      -> sign(x) * 0.5 * log1p(2 * a / (1 - a)),  a = |x|
// Working with a = |x| keeps 1 - a positive (away from 0 except at the |x| == 1
// boundary, handled separately), and the sign is restored at the end.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atanh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::abs(inp);

        // Clamp |x| >= 1 lanes to 0 so the interior formula stays finite there;
        // those lanes are overwritten by the boundary fix-up below.
        v_if(a >= 1.0f) { a = 0.0f; }
        v_endif;

        // Build the log1p argument, then materialise it to DST before the log1p
        // polynomial. Round-tripping through DST cuts the reciprocal->log1p
        // expression so the SFPU register allocator does not exceed its reload
        // budget (the fused form overflows it). The boundary lanes are restored
        // from `inp` afterwards, so clobbering DST here is safe.
        sfpi::vFloat den = 1.0f - a;
        sfpi::dst_reg[0] = (a + a) * _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(den);

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        res = sfpi::copysgn(0.5f * res, inp);

        // Boundary fix-ups: |x| == 1 -> +/-inf, |x| > 1 -> NaN. abs(inp) is
        // recomputed inline here rather than cached in a register; a cached
        // |x| - 1 variant pushed the allocator past the reload budget.
        v_if(sfpi::abs(inp) > 1.0f) { res = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(sfpi::abs(inp) == 1.0f) {
            sfpi::vFloat inf = std::numeric_limits<float>::infinity();
            res = sfpi::copysgn(inf, inp);
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_inverse_hyperbolic() {
    // asinh/acosh route through calculate_log1p_fp32, which expects the log1p
    // polynomial constants in vConstFloatPrgm0/1/2. The sqrt used internally is
    // self-contained (_sfpu_sqrt_ge0_) and does not touch the program registers.
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_atanh() {
    // atanh routes through calculate_log1p_fp32; the reciprocal it uses is the
    // self-contained _sfpu_reciprocal_gt0_, so log1p owns the program registers.
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
