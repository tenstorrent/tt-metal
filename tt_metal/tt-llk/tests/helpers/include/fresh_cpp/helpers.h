// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared semantic helpers for the canonical fresh_cpp/ per-op bodies
// (fresh_cpp/README.md).  Migrated verbatim from fresh_cpp_operations.h
// (Lane BR batches 2/3); consumed by fresh_cpp/<op>.h headers and by the
// aggregator's remaining pre-storm bodies (fmod, softplus, i1) via the
// aggregator's include.  Plain typed C++ only, per the storm contract.
#include <cstdint>

namespace ckernel::sfpu
{

#if __riscv_xtttensixwh
template <int>
struct fresh_recip_supported_on_wh
{
    static constexpr bool value = false;
};
#endif

// Shared: round-to-nearest integer and its int value via the 1.5*2^23
// rounding-bias identity (|z| < 2^22; golden math, the same identity the
// production expm1/exp kernels use through raw bit reads).
sfpi_inline sfpi::vFloat fresh_round_nearest(const sfpi::vFloat z, sfpi::vInt& k_int)
{
    constexpr float ROUNDING_BIAS = 12582912.0f; // 1.5 * 2^23
    const sfpi::vFloat t          = z + ROUNDING_BIAS;
    k_int                         = sfpi::as<sfpi::vInt>(t) - sfpi::as<sfpi::vInt>(sfpi::vFloat(ROUNDING_BIAS));
    return t - ROUNDING_BIAS;
}

// Shared: truncate-toward-zero via round-nearest + downward fixup on the
// magnitude (exact for every finite input; pass-through for |v| >= 2^23,
// inf, NaN — the same contract as the production kernels' exponent-shift
// truncation).
sfpi_inline sfpi::vFloat fresh_trunc_magnitude(const sfpi::vFloat v)
{
    constexpr float MANTISSA_SHIFT = 8388608.0f; // 2^23
    sfpi::vFloat r                 = v;
    sfpi::vFloat t                 = v + MANTISSA_SHIFT;
    t                              = t - MANTISSA_SHIFT;
    v_if (sfpi::exexp(v) < 23)
    {
        r = t;
    }
    v_endif;
    // Nearest may round up; truncation of a non-negative value never does.
    v_if (r > v)
    {
        r = r - 1.0f;
    }
    v_endif;
    return r;
}

// fmod / remainder core (production: metal ckernel_sfpu_fmod.h /
// ckernel_sfpu_remainder.h — divisor and reciprocal smuggled through
// vConstFloatPrgm0/1 by init, exponent-shift truncation, unroll-0 pins).
// Same algorithm: |v| minus trunc(|v|*recip)*s, the fixed residual mop-up,
// and the |v|==s zero snap; divisor/recip are the golden's fixed dispatch
// constants (2.0, 0.5) as plain locals.
sfpi_inline sfpi::vFloat fresh_fmod_core(const sfpi::vFloat v_mag, const sfpi::vFloat s, const sfpi::vFloat recip)
{
    sfpi::vFloat v        = v_mag;
    sfpi::vFloat quotient = fresh_trunc_magnitude(v * recip);
    v                     = v - quotient * s;

    // Residual mop-up (production-identical iteration count; value-bearing).
    constexpr int MOP_UP_ITERATIONS = 10;
    for (int l = 0; l < MOP_UP_ITERATIONS; ++l)
    {
        v_if (v >= s)
        {
            v = v - s;
        }
        v_endif;
    }
    return v;
}

// Shared: exp(x) by the exp_21f exponent/mantissa recombination, clamped form
// (the production _sfpu_exp_21f_bf16_<true> contract: fp32 result, no bf16
// store rounding — callers own the store).  Same golden-math constants as
// calculate_exp_fresh_cpp; kept separate so the measured exp row's fresh body
// stays byte-stable.
sfpi_inline sfpi::vFloat fresh_exp_21f(const sfpi::vFloat val)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr float C0      = 1.0017248f;
    constexpr float C1      = 7.839635491371155e-08f;
    constexpr float C2      = 4.791750143340323e-15f;

    sfpi::vFloat xlog2   = val * ONE_LN2 + 127.0f;
    xlog2                = sfpi::clamp(xlog2, 0.0f, 255.0f);
    const sfpi::vInt zi  = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
    const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
    frac              = (C2 * frac + C1) * frac + C0;
    return sfpi::setexp(frac, sfpi::exexp(z, sfpi::ExponentMode::Biased));
}

// Shared: reciprocal with Newton refinement, all constants literal (the
// production sfpu_reciprocal_iter reads its 2.0 from vConstFloatPrgm0 —
// the hand-ism these bodies remove).  Same NaN-by-sign-check contract.
template <int NEWTON_ITERATIONS>
sfpi_inline sfpi::vFloat fresh_recip(const sfpi::vFloat x)
{
#if __riscv_xtttensixwh
    // SFPARECIP is not available on Wormhole.  Keep the shared semantic
    // header parseable for unrelated WH test cases, but fail at the actual
    // call site if a BH/QSR-only reciprocal body is selected there.
    static_assert(fresh_recip_supported_on_wh<NEWTON_ITERATIONS>::value, "fresh_recip requires BH/QSR SFPARECIP");
    return x;
#else
    sfpi::vFloat y = sfpi::approx_recip(x);
    if constexpr (NEWTON_ITERATIONS > 0)
    {
        sfpi::vFloat t = x * y - 2.0f;
        if constexpr (NEWTON_ITERATIONS > 1)
        {
            const sfpi::vFloat y1 = y * -t - 0.0f;
            v_if (t < 0.0f)
            {
                t = x * y1 - 2.0f;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        }
        else
        {
            v_if (t < 0.0f)
            {
                y = y * -t - 0.0f;
            }
            v_endif;
        }
    }
    return y;
#endif
}

} // namespace ckernel::sfpu
