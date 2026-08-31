// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared golden-math building blocks for the storm-contract semantic bodies
// (fresh_cpp/<op>.h).  These state published, op-independent numeric identities
// in plain typed C++ — no l_reg pinning, raw TTI, replay/macro templates, or
// markers.  They are NOT ops themselves: each op's header states its own
// mathematical definition and only reuses these statements.
#include <cstdint>

namespace ckernel::sfpu
{

#if __riscv_xtttensixwh
template <int>
struct fresh_hwseed_supported_on_wh
{
    static constexpr bool value = false;
};
#endif

// Reciprocal of a strictly positive finite fp32 vector, stated from the
// published bit-pattern seed (Blinn's constant-minus-bits approximation,
// K = 0x7EF127EA; "Floating-point tricks", IEEE CG&A 1997) refined by three
// Newton–Raphson steps r <- r*(2 - x*r).  Each step squares the relative
// error (seed ~1e-1 -> ~1e-8, fp32-limited), so the result is correctly
// rounded to well below every bf16/fp32 tolerance gate in the suite.
sfpi_inline sfpi::vFloat fresh_recip_positive(const sfpi::vFloat x)
{
    constexpr int RECIP_SEED_MAGIC = 0x7EF127EA;
    sfpi::vFloat r                 = sfpi::as<sfpi::vFloat>(sfpi::vInt(RECIP_SEED_MAGIC) - sfpi::as<sfpi::vInt>(x));
    for (int step = 0; step < 3; ++step)
    {
        r = r * (2.0f - x * r);
    }
    return r;
}

// Reciprocal of a nonzero finite fp32 vector, stated from the hardware
// reciprocal estimate (SFPARECIP, ~7 bit seed; sign-agnostic) refined by two
// Newton–Raphson steps.  Each step squares the relative error (~8e-3 ->
// ~6e-5 -> fp32-limited), the same accuracy class as fresh_recip_positive's
// three steps from the ~4.4-bit Blinn seed.  The steps are spelled through
// the identity y*(2 - x*y) = y*(-t) - 0 with t = x*y - 2 so each half is a
// single fused multiply-add: 5 issue slots at dependency depth 5 versus the
// bit-seed statement's 13 slots at depth 11 (lane DJ loser-attack anatomy,
// 2026-08-20).  Callers whose domain includes zero or infinite divisors must
// state those cases themselves (the mod bodies overwrite them explicitly);
// this statement is exact only on nonzero finite lanes.
#if __riscv_xtttensixwh
template <int ARCH_REQUIRES_SFPARECIP = 0>
#endif
sfpi_inline sfpi::vFloat fresh_recip_hwseed(const sfpi::vFloat x)
{
#if __riscv_xtttensixwh
    // SFPARECIP is not available on Wormhole.  Making the rejection
    // template-dependent lets aggregate headers compile for unrelated WH
    // operations while still refusing any selected unsupported body.
    static_assert(fresh_hwseed_supported_on_wh<ARCH_REQUIRES_SFPARECIP>::value, "fresh_recip_hwseed requires BH/QSR SFPARECIP");
    return x;
#else
    sfpi::vFloat y = sfpi::approx_recip(x);
    for (int step = 0; step < 2; ++step)
    {
        const sfpi::vFloat t = x * y - 2.0f;
        y                    = y * -t - 0.0f;
    }
    return y;
#endif
}

// Truncate a non-negative fp32 vector to its integer part: the 2^23
// mantissa-shift round-to-nearest, then take back the one-off when nearest
// rounded up.  Values at or above 2^23 carry no fraction and pass through.
sfpi_inline sfpi::vFloat fresh_trunc_nonneg(const sfpi::vFloat v)
{
    constexpr float MANTISSA_ONE = 8388608.0f; // 2^23
    sfpi::vFloat r               = v;
    const sfpi::vFloat nearest   = (v + MANTISSA_ONE) - MANTISSA_ONE;
    v_if (v < MANTISSA_ONE)
    {
        r = nearest;
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

// Residue of a non-negative dividend modulo a positive divisor:
// r = aa - trunc(aa * (1/ab)) * ab, followed by the two one-step corrections
// that put a rounding-perturbed quotient back into the mathematical range
// 0 <= r < ab.  recip is the divisor's reciprocal (either shared statement).
sfpi_inline sfpi::vFloat fresh_mod_positive(const sfpi::vFloat aa, const sfpi::vFloat ab, const sfpi::vFloat recip)
{
    const sfpi::vFloat quotient = fresh_trunc_nonneg(aa * recip);
    sfpi::vFloat r              = aa - quotient * ab;
    // A quotient one off in either direction leaves r one divisor outside
    // the range; two guarded steps restore 0 <= r < ab.
    v_if (r < 0.0f)
    {
        r = r + ab;
    }
    v_endif;
    v_if (r >= ab)
    {
        r = r - ab;
    }
    v_endif;
    v_if (r < 0.0f)
    {
        r = r + ab;
    }
    v_endif;
    return r;
}

// exp(x) - 1 stated by the Cody-Waite reduction the production elu/celu/selu
// family shares (tt_llk ckernel_sfpu_expm1_cw.h expm1_cw_clamped, bf16 arm):
//   x = k*ln2 + r with |r| <= ln2/2; expm1(r) = r * h(r) with h the published
//   Sollya remez degree-4 fit (max abs error 1.60e-7); reconstruction
//   expm1(x) = (2^k - 1) + 2^k * expm1(r).
// k comes from the 1.5*2^23 rounding-bias trick: the biased-integer reading of
// t = x/ln2 + BIAS carries k in its low bits, and 2^k is rebuilt with setexp.
// The range reduction is also the tt-polynomial-fitter deployment canon
// (deployment/.../piecewise_generic.cpp exp_reduce: identical INV_LN2 /
// NEG_LN2_HI / NEG_LN2_LO split and rounding-bias lowering); the h(r)
// coefficients are the production kernel's own Sollya fit (not in the fitter's
// data).  Expression shapes mirror the production body statement-for-statement
// so both legs lower to the same value stream (lane JU bit-exact repair: the
// prior exp-recombination arm gave elu(0) = +1.72e-3, lane JN certificate).
// The clamp at -87 keeps k >= -126 so the setexp reconstruction cannot wrap.
sfpi_inline sfpi::vFloat fresh_expm1_cw(sfpi::vFloat x)
{
    constexpr float CW_INV_LN2    = 1.4426950408889634f;
    constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
    constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;
    constexpr float ROUNDING_BIAS = 12582912.0f; // 1.5 * 2^23
    // h(r) = expm1(r)/r on [-ln2/2, ln2/2], ascending powers (production fit).
    constexpr float H1 = 4.9999371171e-01f;
    constexpr float H2 = 1.6666433215e-01f;
    constexpr float H3 = 4.1875664145e-02f;
    constexpr float H4 = 8.3751315251e-03f;

    x = sfpi::max(x, -87.0f);

    // Cody-Waite range reduction: x = k*ln2 + r (keep the multiply fused into
    // the bias add, exactly as the production body spells it).
    const sfpi::vFloat bias = ROUNDING_BIAS;
    const sfpi::vFloat t    = x * CW_INV_LN2 + bias;
    const sfpi::vFloat k    = t - bias;
    sfpi::vFloat r          = k * CW_NEG_LN2_HI + x;
    r                       = r + k * CW_NEG_LN2_LO;

    // expm1(r) = r * h(r), Horner.
    sfpi::vFloat h = H4 * r + H3;
    h              = h * r + H2;
    h              = h * r + H1;
    h              = h * r + 1.0f;
    h              = r * h;

    // 2^k from the biased-integer reading of t; expm1(x) = (2^k - 1) + 2^k*h.
    const sfpi::vInt k_int   = sfpi::as<sfpi::vInt>(t) - sfpi::as<sfpi::vInt>(bias);
    const sfpi::vFloat two_k = sfpi::setexp(sfpi::vFloat(1.0f), k_int + 127);
    return (two_k - 1.0f) + two_k * h;
}

// exp(x) stated by exponent/mantissa recombination (the exp_21f algorithm of
// Moroz, Samotyy, Walczyk & Cieslinski 2022): exp(x) = 2**(x/ln2) = 2**xi *
// 2**xf; xlog2 = x/ln2 + 127 is the result's biased exponent, its fixed-point
// encoding is split into exponent (integer) and mantissa (fraction) fields,
// the fraction is refined by the published quadratic, and the two recombine
// with setexp.  Overflow saturates the biased exponent at 255; underflow
// (xlog2 <= 0) zeroes the exponent source so the recombination produces a
// subnormal that the bf16 store path flushes to zero.
sfpi_inline sfpi::vFloat fresh_exp(const sfpi::vFloat x)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr float C0      = 1.0017248f;
    constexpr float C1      = 7.839635491371155e-08f;
    constexpr float C2      = 4.791750143340323e-15f;

    sfpi::vFloat xlog2 = x * ONE_LN2 + 127.0f;
    xlog2              = sfpi::min(xlog2, 255.0f);

    // Fixed-point encoding of xlog2: mantissa (implicit one) shifted left by
    // the unbiased exponent.
    const sfpi::vInt iexp = sfpi::exexp(xlog2);
    sfpi::vInt zi         = sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne);
    zi                    = sfpi::shft(zi, iexp, sfpi::ShiftMode::Logical);
    const sfpi::vFloat z  = sfpi::as<sfpi::vFloat>(zi);

    // Quadratic refinement of 2**xf on [0, 1) from the encoding's mantissa.
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
    frac              = (C2 * frac + C1) * frac + C0;

    // Underflow: zero the exponent source where xlog2 is not positive.
    sfpi::vFloat zc = z;
    v_if (xlog2 <= 0.0f)
    {
        zc = 0.0f;
    }
    v_endif;

    return sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));
}

} // namespace ckernel::sfpu
