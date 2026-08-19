// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the lcm corpus row (storm contract — see
// README.md in this directory).  Include from an LLK_TRISC_MATH kernel after
// sfpu_operations.h; plain typed C++ only.

#include <cstdint>

namespace ckernel::sfpu
{

// ctz(v) = 31 - clz(isolated lowest set bit); every lane that consumes the
// result holds v != 0 (the callers predicate on it).
inline sfpi::vInt lcm_fresh_cpp_ctz(const sfpi::vInt v)
{
    const sfpi::vInt iso = v & (sfpi::vInt(0) - v);
    return sfpi::vInt(31) - sfpi::as<sfpi::vInt>(sfpi::lz(iso));
}

// Fresh typed-C++ lcm stating the torch.lcm golden contract independently,
// one datum per row over the binary harness's adjacent-tile Int32 Dst:
//
//   lcm(a, b) = (|a| / gcd(|a|, |b|)) * |b|   (non-negative, exact Int32)
//
// The production kernel (metal ckernel_sfpu_lcm.h) is a raw hand-scheduled
// TTI stream: fixed LREG0..5 allocation, a REPLAY-buffered binary-GCD body
// shared with gcd, SETSGN/SETEXP reciprocal normalization with two seed
// constants parked in vConstFloatPrgm0/1, and an SFPMUL24 pair for the final
// product.  This body states the same mathematics in plain portable typed
// SFPI — the compiler owns allocation, scheduling, replay, and delivery:
//
//   1. gcd by the binary subtract-shift algorithm, 15 predicated rounds
//      (the production gcd body's own bound: worst case for n-bit inputs is
//      n rounds; the row's domain is 1 <= a, b <= 20000 < 2^15, the same
//      |a|, |b| < 2^15 precondition the production kernel documents);
//   2. q = |a| / g through a Newton reciprocal from the classic 48/17 -
//      (32/17)*m seed on m in [0.5, 1) (rel. err <= 1/17; three refinements
//      leave |q_float - q| far below the 0.5 the nearest-integer recovery
//      needs), recovered exactly with the 2^23 round-and-extract identity;
//   3. q * |b| split as q*(b >> 8) and q*(b & 0xFF): with q, |b| < 2^15 both
//      partial products stay below 2^23, so fp32 arithmetic and the same
//      2^23 extraction are exact, and the 30-bit product is recombined in
//      integer space.
//
// Every constant is a plain local; no fixed LREGs, raw instructions, replay
// slots, or SFPLOADMACRO templates.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_lcm_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    constexpr float BIAS              = 8388608.0f; // 2^23: round-and-extract anchor
    // Newton reciprocal seed for m in [0.5, 1): r0 = 48/17 - (32/17)*m.
    constexpr float SEED_C0 = 48.0f / 17.0f;
    constexpr float SEED_C1 = 32.0f / 17.0f;

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            // Keep the all-lane predicate boundary typed and local (the fresh
            // binary max/min precedent).
            __builtin_rvtt_sfppushc(0);
            __builtin_rvtt_sfppopc(0);
            const sfpi::vInt a  = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b  = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt ax = sfpi::as<sfpi::vInt>(sfpi::abs(a));
            const sfpi::vInt bx = sfpi::as<sfpi::vInt>(sfpi::abs(b));

            // --- 1. g = gcd(ax, bx), binary subtract-shift form.
            // Invariant: gcd(x, y) * 2^common is the answer; x is kept odd,
            // each round makes y odd, orders x <= y, and subtracts, so y
            // turns even (or 0) and loses at least one bit per round.
            sfpi::vInt x            = ax;
            sfpi::vInt y            = bx;
            const sfpi::vInt common = lcm_fresh_cpp_ctz(x | y);
            x                       = sfpi::shft(x, sfpi::vInt(0) - lcm_fresh_cpp_ctz(x), sfpi::ShiftMode::Logical);
            for (int round = 0; round < 15; ++round)
            {
                v_if (y != 0)
                {
                    y = sfpi::shft(y, sfpi::vInt(0) - lcm_fresh_cpp_ctz(y), sfpi::ShiftMode::Logical);
                    v_if (x > y)
                    {
                        const sfpi::vInt t = x;
                        x                  = y;
                        y                  = t;
                    }
                    v_endif;
                    y = y - x;
                }
                v_endif;
            }
            const sfpi::vInt g = sfpi::shft(x, common, sfpi::ShiftMode::Logical);

            // --- 2. q = ax / g (exact: g divides ax).
            // Reciprocal of g: normalize to m in [0.5, 1), Newton-refine the
            // linear seed, then rebias the exponent by the normalization
            // shift.  g < 2^15 converts to fp32 exactly.
            const sfpi::vFloat gf = sfpi::convert<sfpi::vFloat>(g, sfpi::RoundMode::Nearest);
            const sfpi::vFloat gm = sfpi::setexp(gf, 126); // m in [0.5, 1)
            sfpi::vFloat rm       = SEED_C0 - SEED_C1 * gm;
            rm                    = rm * (2.0f - gm * rm);
            rm                    = rm * (2.0f - gm * rm);
            rm                    = rm * (2.0f - gm * rm);
            // recip(g) = rm * 2^(126 - biased_exp(g)).
            const sfpi::vInt rebias = sfpi::exexp(rm, sfpi::ExponentMode::Biased) + (sfpi::vInt(126) - sfpi::exexp(gf, sfpi::ExponentMode::Biased));
            const sfpi::vFloat rg   = sfpi::setexp(rm, rebias);

            // Nearest-integer recovery: q_float + 2^23 rounds to the exact
            // integer q in the mantissa field (q < 2^15, error << 0.5).
            const sfpi::vFloat af = sfpi::convert<sfpi::vFloat>(ax, sfpi::RoundMode::Nearest);
            const sfpi::vInt q    = sfpi::as<sfpi::vInt>(sfpi::exman(af * rg + BIAS));
            const sfpi::vFloat qf = sfpi::convert<sfpi::vFloat>(q, sfpi::RoundMode::Nearest);

            // --- 3. result = q * bx, exact through an 8-bit split.
            const sfpi::vFloat bhi = sfpi::convert<sfpi::vFloat>(sfpi::shft(bx, -8, sfpi::ShiftMode::Logical), sfpi::RoundMode::Nearest);
            const sfpi::vFloat blo = sfpi::convert<sfpi::vFloat>(bx & 0xFF, sfpi::RoundMode::Nearest);
            const sfpi::vInt p_hi  = sfpi::as<sfpi::vInt>(sfpi::exman(qf * bhi + BIAS));
            const sfpi::vInt p_lo  = sfpi::as<sfpi::vInt>(sfpi::exman(qf * blo + BIAS));
            const sfpi::vInt lcm   = sfpi::shft(p_hi, 8, sfpi::ShiftMode::Logical) + p_lo;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = lcm;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_lcm_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh lcm expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh lcm expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh lcm expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic
    // body contains only constant relative Dst addresses (the fresh binary
    // max/min, add/sub, and mul precedent).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_lcm_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
