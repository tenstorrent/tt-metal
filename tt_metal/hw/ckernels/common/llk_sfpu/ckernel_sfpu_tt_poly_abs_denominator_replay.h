// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Canonical selected replay arithmetic. Callers own replay count and DST lifecycle.
namespace sfpi {
#if defined(ARCH_BLACKHOLE)
inline void abs_denominator_loadmacro_init() {
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

template <uint32_t Hold, uint32_t Advance>
inline void abs_denominator_bh_body() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // x
    TTI_SFPNOP;                                       // conservative replay load-consumer gap
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                  1);  // magnitude = |x|
    TTI_SFPSWAP(
        0,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // magnitude=min(|x|, bound)
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // den = magnitude + 1.0
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_neg1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        0);                     // finite: 0 exactly; decoded +/-Inf: NaN; den hazard gap
    TTI_SFPARECIP(0, 7, 2, 0);  // reciprocal seed -> L2
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                  0);  // restore x sign; reciprocal-seed hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG3,
        2);  // t = den*r0 - 2.0
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        0);  // finite identity / exponent-FF poison fills residual hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0);  // CC: first residual < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        1);  // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        2);                    // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // CC-balanced replay body
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // selected y = bounded_x * reciprocal
    // The load-macro itself is the result-MUL hazard gap.  Captured T1 RNE
    // fires at issue+1, then the fixed store advances the same destination
    // row at issue+3. L4 is dead after the second Newton update.
    TTI_SFPLOADMACRO((0 << 2) | (ckernel::p_sfpu::LREG4 & 3), 0, Advance, (ckernel::p_sfpu::LREG4 >> 2));
}
#elif defined(ARCH_WORMHOLE)
template <uint32_t Slopes, uint32_t Intercepts>
inline void abs_denominator_lut_init() {
    TTI_SFPLOADI(0, SFPLOADI_MOD0_USHORT, 0x80ff);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(1, 2, Slopes & 0xffff);
    TTI_SFPLOADI(1, 8, Slopes >> 16);
    TTI_SFPLOADI(5, 2, Intercepts & 0xffff);
    TTI_SFPLOADI(5, 8, Intercepts >> 16);
}

template <uint32_t BoundExponent, bool LateRound, uint32_t Hold>
inline void abs_denominator_wh_core() {
    TTI_SFPLOAD(4, 0, Hold, 0);
    TTI_SFPLOAD(7, SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    TTI_SFPSETSGN(0, 4, 3, 1);
    TTI_SFPADDI(0x3f80, 3, 0);
    TTI_SFPEXEXP(0, 4, 6, SFPEXEXP_MOD1_NODEBIAS);
    TTI_SFPEXEXP(0, 3, 2, 0);
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LCONST_neg1, 3, 0);
    TTI_SFPIADD(0, 6, 2, 6);
    TTI_SFPSETEXP(0, 4, 2, 0);
    TTI_SFPLUTFP32(0, 2);
    TTI_SFPXOR(0, 12, 7, 0);
    TTI_SFPMUL(2, 0, ckernel::p_sfpu::LCONST_0, 4, 0);
    TTI_SFPMAD(3, 0, ckernel::p_sfpu::LCONST_1, 2, 0);
    TTI_SFPIADD((-255) & 0xfff, 6, 3, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(4, 2, 4, 4, 0);
    if constexpr (LateRound) {
        TTI_SFPIADD((-BoundExponent) & 0xfff, 6, 6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);
    } else {
        TTI_SFPIADD((-BoundExponent) & 0xfff, 6, 6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 4, 4, 4, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSETCC(0, 6, 0, SFPSETCC_MOD1_LREG_GTE0);
    }
}

template <bool LateRound, uint32_t Advance>
inline void abs_denominator_wh_suffix() {
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LCONST_1, 4, 0);
    TTI_SFPSETCC(0, 3, 0, SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, 0x7f80);
    TTI_SFPSETCC(0, 7, 0, SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, 0xff80);
    TTI_SFPENCC(0, 0, 0, 0);
    if constexpr (LateRound) {
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 4, 4, 4, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }
    TTI_SFPSTORE(4, 0, Advance, 0);
}
#endif
}  // namespace sfpi
