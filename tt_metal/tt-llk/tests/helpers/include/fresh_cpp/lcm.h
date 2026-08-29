// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// lcm — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// States the torch.lcm golden contract (golden_generators._lcm, exact Int32)
// over the row's stimulus domain (test_sfpu_binary._INT_BINARY_STIMULI
// [SfpuLcm] = uniform [1, 20000], both operands < 2^15):
//
//   lcm(a, b) = (|a| / gcd(|a|, |b|)) * |b|   (non-negative, exact Int32)
//
// built from three certificate-backed pieces (the gcd-v2 rewrite skeleton,
// fresh_cpp/gcd.h):
//
// 1. gcd as UNPREDICATED fixed-point Stein rounds — strip b's trailing
//    zeros, sort, subtract — with NO per-round v_if/setcc/encc.  Termination
//    is a fixed point instead of a predicate: (g, 0) -> (0, g) and (0, g)
//    is a fixed point of the round (ctz's lz(0) = 32 spelling shifts 0 by
//    one bit keeping 0; the sort puts 0 first; the subtract returns g).
//    Round bound: 15, exhaustively certified over the whole [1, 20000]^2
//    domain (laneDI-evidence-20260820/lcm_cert.c: DP over all odd pairs,
//    max rounds-to-zero = 15, witness (16383, 16385) = 2^14 -+ 1;
//    forward-sound because strip and subtract never increase a lane's max
//    component).  The 15th round is TRIMMED to strip + sort: for every
//    lane, x + y after a full round equals the sort's max, and the
//    certificate makes y = 0 after round 15, so round 15's sorted max IS
//    the odd gcd — the final subtract and the a+b recombination both fold
//    away (the hand kernel's own last-round trim, restated as semantics).
//    The loop CARRIES -y instead of y (the hand kernel's {-a, a} register
//    pair, restated as semantics): the subtract is stated as min - max
//    (exact, negative), the next round's integer abs re-derives the
//    magnitude, and y & -y needs no per-round re-negation — every
//    destructive lane op lands on a value that just died, so the round is
//    movs-free.  (0, -g) is the fixed point; abs(0) = 0 keeps the zero
//    walk intact.
// 2. q = (|a| >> c) / g_odd where c = ctz(|a| | |b|) is the common power
//    of two: g = g_odd << c divides |a|, and c <= ctz(|a|), so the shifted
//    dividend stays an exact multiple of g_odd and the common shift is
//    consumed entirely in the prologue (no epilogue un-shift).  The
//    reciprocal is the magic-constant seed (0x7EF311C3 - bits, the
//    fresh_cpp/digamma.h fresh_recip_positive_blinn constant of record)
//    with two Newton refinements in fused a*b+c form, r <- r + r*(1 - g*r),
//    against the one-time negated divisor (vConst1 keeps every refinement
//    constant in a constant register); no normalize/rebias step.  The
//    nearest-integer recovery is the 2^23 round-and-extract identity
//    (q < 2^23; the fresh div_int32_floor precedent).  Exhaustively
//    certified over every reachable (dividend, divisor) pair of the domain
//    and every fused/unfused rounding assignment of the five a*b+c sites,
//    including bit-exact BH SFPMAD semantics (craq fma_model_bh):
//    max |float_quotient - q| = see lcm_cert.out, margin far below the 0.5
//    the recovery needs (laneDI-evidence-20260820/lcm_cert.c).
// 3. result = q * |b| through the typed 24x24 integer-multiply primitive
//    (sfpi::fractional_mul, the fresh mul-int precedent): q, |b| < 2^15,
//    so the product < 2^30 splits exactly as low-23-bits + (high << 23)
//    with no >= 2^23 correction terms.
//
// The ordering step is the typed min/max sort (one architectural SFPSWAP);
// all sorted operands are non-negative, where sign-magnitude and integer
// order coincide (the gcd-v2 fact: sign_mag32_total_order == integer order
// for ALL non-negative values).  abs() of both operands is kept — the
// production kernel documents it and torch.lcm is |.|-symmetric — and the
// non-negative results make every as<vSMag> reinterpretation exact.
// Production: metal ckernel_sfpu_lcm.h (hand-issued REPLAY-loop kernel).
// The previous 15-round predicated body is preserved in
// fresh_cpp/lcm_legacy.h.
#include <cstdint>

namespace ckernel::sfpu
{

#if __riscv_xtttensixwh
template <int>
struct fresh_lcm_supported_on_wh
{
    static constexpr bool value = false;
};
#endif

// Right-shift amount that drops v's trailing zeros: -ctz(v) = clz(isolated
// lowest set bit) - 31 (negative = logical right shift for sfpi::shft).
// Stated as the shift amount directly so the consumer needs no re-negation
// and the 31 stays an sfpiadd immediate.  For v == 0 the isolated bit is 0,
// lz(0) = 32 gives +1 (a one-bit left shift), and 0 << 1 == 0 — zero is
// preserved, which the fixed-point rounds below rely on.
sfpi_inline sfpi::vInt lcm_fresh_cpp_ctz_shift(const sfpi::vInt v)
{
    const sfpi::vInt iso = v & (sfpi::vInt(0) - v);
    return sfpi::as<sfpi::vInt>(sfpi::lz(iso)) - 31;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_lcm_fresh_cpp()
{
#if __riscv_xtttensixwh
    // This semantic body uses BH/QSR's 24x24 fractional multiply.  Preserve
    // an explicit instantiation-time refusal on WH without making every
    // unrelated binary test TU fail while parsing this aggregate header.
    static_assert(fresh_lcm_supported_on_wh<ITERATIONS>::value, "fresh LCM requires BH/QSR SFPMUL24");
#else
    constexpr std::uint32_t tile_rows = 32;
    // Certified round bound for the [1, 20000]^2 stimulus domain: header.
    constexpr int LCM_GCD_ROUNDS = 15;
    // 2^23: round-and-extract anchor for the nearest-integer recovery.
    constexpr float BIAS = 8388608.0f;

    // Park the reciprocal magic seed in a programmable constant register
    // once per call (the production kernel's own init parks its Newton
    // constants in vConstFloatPrgm0/1 the same way): the seed subtraction
    // below reads it as an LREG operand instead of re-materializing the
    // 32-bit immediate with a paired SFPLOADI every row (laneDX: -2
    // issue/exec words per row vs the measured pin-14 stream).
    sfpi::vConstIntPrgm0 = 0x7EF311C3;

    for (int face = 0; face < 4; ++face)
    {
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt a  = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt b  = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::SM32>();
            const sfpi::vInt ax = sfpi::as<sfpi::vInt>(sfpi::abs(a));
            const sfpi::vInt bx = sfpi::as<sfpi::vInt>(sfpi::abs(b));

            // Common power of two c = ctz(ax | bx), consumed immediately:
            // axs = ax >> c is the exact multiple of g_odd the quotient
            // divides (header piece 2), so nothing about c survives the
            // round loop.  Then keep x odd.
            const sfpi::vInt axs = sfpi::shft(ax, lcm_fresh_cpp_ctz_shift(ax | bx), sfpi::ShiftMode::Logical);
            sfpi::vInt x         = sfpi::shft(ax, lcm_fresh_cpp_ctz_shift(ax), sfpi::ShiftMode::Logical);
            // The loop carries -y (header piece 1): the subtract below lands
            // negative and each round's integer abs re-derives the magnitude.
            sfpi::vInt yn = sfpi::vInt(0) - bx;

            for (int round = 0; round < LCM_GCD_ROUNDS - 1; ++round)
            {
                const sfpi::vInt pos = sfpi::as<sfpi::vInt>(sfpi::abs(yn));           // y
                const sfpi::vInt sh  = sfpi::as<sfpi::vInt>(sfpi::lz(pos & yn)) - 31; // -ctz(y): pos & yn == y & -y
                const sfpi::vInt ys  = sfpi::shft(pos, sh, sfpi::ShiftMode::Logical); // strip; 0 << 1 keeps 0
                const auto ordered   = sfpi::min_max(sfpi::as<sfpi::vSMag>(x), sfpi::as<sfpi::vSMag>(ys));
                x                    = sfpi::as<sfpi::vInt>(ordered.first);
                yn                   = x - sfpi::as<sfpi::vInt>(ordered.second); // min - max <= 0, exact
            }
            // Trimmed round 15: strip + sort only; the sorted max is the odd
            // gcd (header piece 1).
            const sfpi::vInt pos = sfpi::as<sfpi::vInt>(sfpi::abs(yn));
            const sfpi::vInt sh  = sfpi::as<sfpi::vInt>(sfpi::lz(pos & yn)) - 31;
            const sfpi::vInt ys  = sfpi::shft(pos, sh, sfpi::ShiftMode::Logical);
            const auto ordered   = sfpi::min_max(sfpi::as<sfpi::vSMag>(x), sfpi::as<sfpi::vSMag>(ys));
            const sfpi::vInt g   = sfpi::as<sfpi::vInt>(ordered.second); // g_odd

            // q = axs / g_odd (exact).  Both convert to fp32 exactly
            // (< 2^15; non-negative, so the 2's-complement bits ARE the
            // sign-magnitude bits the cast consumes).
            const sfpi::vFloat gf = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(g), sfpi::RoundMode::Nearest);
            const sfpi::vFloat af = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(axs), sfpi::RoundMode::Nearest);

            // Magic-constant reciprocal seed + two Newton refinements in
            // a*b+c form against the once-negated divisor (header piece 2).
            sfpi::vFloat r        = sfpi::as<sfpi::vFloat>(sfpi::vInt(sfpi::vConstIntPrgm0) - sfpi::as<sfpi::vInt>(gf));
            const sfpi::vFloat ng = -gf;
            r                     = (ng * r + sfpi::vConst1) * r + r;
            r                     = (ng * r + sfpi::vConst1) * r + r;

            // Nearest-integer recovery: q + 2^23 lands the exact quotient in
            // the mantissa field (q < 2^23, certified margin << 0.5).
            const sfpi::vInt q = sfpi::as<sfpi::vInt>(sfpi::exman(af * r + BIAS));

            // result = q * bx through the typed 24x24 primitive (header
            // piece 3).
            const sfpi::vInt p_lo = sfpi::fractional_mul(q, bx, sfpi::FractionalHalf::Low);
            const sfpi::vInt p_hi = sfpi::fractional_mul(q, bx, sfpi::FractionalHalf::High);
            const sfpi::vInt lcm  = sfpi::shft(p_hi, 23, sfpi::ShiftMode::Logical) + p_lo;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = lcm;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
#endif
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
