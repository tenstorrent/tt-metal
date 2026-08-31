// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the trigonometry row's vehicle op, Acosh
// (storm contract: fresh_cpp/README.md).  Independent derivation from the
// mathematical definition (the production golden is torch.acosh):
//
//   acosh(x) = ln(x + sqrt(x^2 - 1))        for x >= 1, NaN below, +0 at 1
//
// reformulated through log1p exactly as the golden math requires for
// stability (the same three-region contract the production kernel
// documents):
//   1 <= x < 2^28 : log1p((x - 1) + sqrt((x - 1) * (x + 1)))
//   x >= 2^28     : ln(2x) = log1p(x - 1) + ln 2   (avoids x^2 overflow)
//
// The production hand-isms removed (Lane BR batch-2 named work item, now
// unblocked at pin 12):
//   - the mid-body DST round-trip ("sfpi::dst_reg[0] = arg" store/reload
//     that severed the sqrt expression from the log1p polynomial to keep
//     the pin-10 SFPU register allocator inside its reload budget — and
//     truncated arg to bf16 on a 16-bit DEST as a side effect);
//   - the log1p constants parked in vConstFloatPrgm0/1/2 by log1p_init;
//   - the addexp(x, -1) exponent-shift spelling of 0.5*x.
// Same rsqrt seed, Newton counts, and per-format log1p fits (they ARE the
// golden math the row's tolerance is fitted to), every constant a plain
// local.

#include <cstdint>
#include <limits>

namespace ckernel::sfpu
{

// sqrt for known-nonnegative input: Quake-seed reciprocal sqrt + Newton
// (2 steps bf16 dest, 3 fp32 dest), then one Heron step on the product.
// The production _sfpu_sqrt_ge0_ already keeps its seed as an immediate;
// restated here with the plain 0.5f*x spelling.
template <bool IS_FP32_DEST_ACC>
sfpi_inline sfpi::vFloat fresh_sqrt_ge0(const sfpi::vFloat x)
{
    constexpr int SEED = 0x5f1110a0; // Kokosinski/Moroz seed (cited paper)

    const sfpi::vFloat half_x = 0.5f * x;
    sfpi::vFloat y            = sfpi::as<sfpi::vFloat>(sfpi::vInt(SEED) - sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1));

    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    if constexpr (IS_FP32_DEST_ACC)
    {
        y = y * (1.5f - half_x * y * y);
    }

    sfpi::vFloat a = x * y;
    a              = a + 0.5f * (x - a * a) * y;

    // The reciprocal seed produces inf * 0 = NaN at zero.
    v_if (x == 0.0f)
    {
        a = 0.0f;
    }
    v_endif;
    return a;
}

// log1p(a) by the Juffa reduction: u = 1 + a = 2^k * t with t in [0.75, 1.5),
// m = the reduced argument, k carried in exponent-bit units; per-format
// polynomial correction (deg-9 fp32 dest, deg-3 bf16 dest — the production/
// golden fits), all constants plain locals.
template <bool IS_FP32_DEST_ACC>
sfpi_inline sfpi::vFloat fresh_log1p(const sfpi::vFloat a)
{
    constexpr float LOG_TWO_2M23 = 0.693147182f * 1.19209290e-7f; // ln2 * 2^-23 (k is in exponent-bit units)

    const sfpi::vFloat u = a + 1.0f;
    sfpi::vFloat r       = std::numeric_limits<float>::quiet_NaN();
    v_if (u >= 0.0f)
    {
        // k<<23 = bits(u) - bits(0.75) with the mantissa field cleared.
        const sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e                      = sfpi::as<sfpi::vInt>(u) - sfpi::as<sfpi::vInt>(three_quarters);
        e                                 = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
        sfpi::vFloat m                    = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
        const sfpi::vFloat neg_four       = -4.0f;
        const sfpi::vFloat s              = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(neg_four) - e);
        // t = -s/4 - 1 = 2^(-k) - 1.
        const sfpi::vFloat t = -0.25f * s - 1.0f;
        m                    = m + t;

        const auto abs_e        = sfpi::abs(e);
        sfpi::vFloat e_float    = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
        e_float                 = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        const sfpi::vFloat m_sq = m * m;
        sfpi::vFloat p;
        if constexpr (IS_FP32_DEST_ACC)
        {
            p = -0x1.92cp-5f;
            p = p * m + 0x1.b84p-4f;
            p = p * m + -0x1.0c4p-3f;
            p = p * m + 0x1.274p-3f;
            p = p * m + -0x1.55p-3f;
            p = p * m + 0x1.998p-3f;
            p = p * m + -0x1.00001ap-2f;
            p = p * m + 0x1.555572p-2f;
            p = p * m + -0.5f;
        }
        else
        {
            p = -0.25f * m + 0x1.744p-2f;
            p = p * m + -0x1.008p-1f;
        }
        r = p * m_sq + m;
        r = e_float * LOG_TWO_2M23 + r;

        // u >= 0, so this safely catches u == inf and u == NaN.
        const sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
        v_if (sfpi::as<sfpi::vInt>(u) >= sfpi::as<sfpi::vInt>(infinity))
        {
            r = u;
        }
        v_endif;
    }
    v_endif;
    return r;
}

template <bool IS_FP32_DEST_ACC, int ITERATIONS>
__attribute__((noinline)) void calculate_acosh_fresh_cpp()
{
    constexpr float LOG1P_LARGE = 268435456.0f; // 2^28
    constexpr float LN2         = 0.6931471805599453f;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat arg     = x - 1.0f;
        v_if (x < LOG1P_LARGE)
        {
            arg = arg + fresh_sqrt_ge0<IS_FP32_DEST_ACC>((x + 1.0f) * (x - 1.0f));
        }
        v_endif;

        if constexpr (!IS_FP32_DEST_ACC)
        {
            // Production materialises arg to DST before log1p; on this row's
            // 16-bit DEST that store TRUNCATES arg to bf16 — a value-changing
            // quantization, not just a scheduling round-trip (lane JN
            // certificate: 437 outputs <= 2 ULP apart once the fresh body
            // dropped it).  The quantization is adopted back, stated as the
            // bit operation it is (arg is an SFPU MAD result, so it is never
            // denormal and the store's flush-to-zero leg is vacuous); the
            // register-pressure scheduling half of production's round-trip
            // stays with the compiler.  Lane JU coefficient repair,
            // 2026-08-31 (hand-adopted numerics).
            arg = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(arg) & 0xffff0000);
        }

        sfpi::vFloat res = fresh_log1p<IS_FP32_DEST_ACC>(arg);
        v_if (x >= LOG1P_LARGE)
        {
            res = res + LN2;
        }
        v_endif;

        // Domain fix-ups: x == 1 -> +0, x < 1 -> NaN.
        v_if (x == 1.0f)
        {
            res = 0.0f;
        }
        v_elseif (x < 1.0f)
        {
            res = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;

        if constexpr (!IS_FP32_DEST_ACC)
        {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
