// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Selected product bodies. Caller owns initialization and SFPU start/done.
namespace sfpi {
inline void product_symmetric_load_macro_init() {
    // Reuse the common raw-ingress RNE/store LOADMACRO carrier.  The
    // macro's load into dead L2 is overwritten by the next replay's
    // absolute-value head; RNE reads literal L0 into staging and the
    // fixed store advances ADDR_MOD_6 at issue+3.
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    constexpr uint32_t kSymmetricProductLmSequence = 0x13850000u;
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (kSymmetricProductLmSequence >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (kSymmetricProductLmSequence & 0xFFFFu));
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
    TTI_SFPNOP;
}
template <class Config, uint32_t Hold, uint32_t Advance>
inline void product_symmetric_bh() {
    constexpr uint32_t kSymMultBits = __builtin_bit_cast(uint32_t, (float)Config::kMultiplier);
    constexpr uint32_t kSymC0Bits = __builtin_bit_cast(uint32_t, Config::kCoefficients[0]);
    constexpr int kSymRecordedBodyLenD2 = 23;

    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSymMultBits >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSymMultBits & 0xffffu));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, (kSymC0Bits >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, (kSymC0Bits & 0xffffu));
    TTI_REPLAY(0, kSymRecordedBodyLenD2, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 0);  // x
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_FLOATB,
        Config::kBiasBf16);  // exact bias; load hazard filler
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2,
                  1);  // |x|; preserved through the replay suffix
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG7,
        0);  // fused biased xlog2
    TTI_SFPNOP;
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, 5, 0, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 1);
    TTI_SFPCAST(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG6, 0);
    TTI_SFPGT(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 8);
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPAND(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 1);
    TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0,
                  2);  // biased exp polynomial y
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        0);  // x+|x|; L1 keeps raw x through the suffix
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        0);                                          // |x|*y
    TTI_SFPMULI(0x3780, ckernel::p_sfpu::LREG0, 0);  // y*2^-16
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        0);  // 1+y*2^-16
    TTI_SFPARECIP(0, 0, 5, 0);
    TTI_SFPMULI(0x3800, ckernel::p_sfpu::LREG6, 0);  // |x|*y*2^-15
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 2);
    TTI_SFPNOP;
#pragma GCC unroll 32
    for (int d = 0; d < 32; d++) {
        if (d != 0) {
            TTI_REPLAY(0, kSymRecordedBodyLenD2, 0, 0);
        }
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 3);
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG0,
            1);  // doubled signed result
        // Fill the result-MAD -> exact-half dependency window
        // with the independent typed origin-half discriminator.
        // Scaling |x| by 2^126 makes the top first-binade magnitude
        // ordinary 1.9921875, so the test survives FP32 FTZ arithmetic.
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG5, 1);
        // Clamp the doubled result to raw x before halving.  Besides
        // the ordinary lower bound, this owns the positive 0x0100
        // origin word that cannot survive a direct FP32 half under FTZ.
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMULI(0x3F00, ckernel::p_sfpu::LREG0, 0);  // exact half
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(
            0,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            1);  // raw-signed MIN_NORMAL for the exact half tie
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(
            0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG0,
            1);  // positive identity cap; negative values pass through
        TTI_SFPLOADMACRO(2, 0, Advance, 0);
    }
    // The final macro's fixed store fires at issue+3.  Other elements
    // drain under the next replay head; only the last needs padding.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
template <class Config, uint32_t Hold, uint32_t Advance>
inline void product_bounded_bh() {
    constexpr uint32_t kMultBits = __builtin_bit_cast(uint32_t, (float)Config::kMultiplier);
    constexpr uint32_t kBiasBits = __builtin_bit_cast(uint32_t, (float)Config::kBias);
    constexpr uint32_t kC0Bits = __builtin_bit_cast(uint32_t, Config::kCoefficients[0]);
    constexpr uint32_t kNegSlopeBits = __builtin_bit_cast(uint32_t, (float)Config::kNegativeSlope);
    constexpr uint32_t kNegativePositiveScaleBits = __builtin_bit_cast(uint32_t, -(float)Config::kPositiveScale);

    auto restore_cross_row_pins = [&]() {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kMultBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kMultBits & 0xffffu));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, (kBiasBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (kC0Bits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, (kC0Bits & 0xffffu));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, Config::kCoordinateUpperBf16);
    };
    restore_cross_row_pins();

    TTI_REPLAY(0, 32, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 0);  // 1: x
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        0);  // 2: x*log2(e)+scaled bias
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER,
                 (kNegSlopeBits >> 16));  // 3: producer gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0,
                9);  // 4: max(xlog2, 0)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kNegSlopeBits & 0xffffu));  // 5: lower-clamp retirement gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0,
                1);  // 6: min(xlog2, typed right-bound coordinate)
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_UPPER,
        (kNegativePositiveScaleBits >> 16));                             // 7: clamp retirement gap
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 0);  // 8
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);  // 9
    TTI_SFPSHFT(0, 4, 0, 0);                                             // 10
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                 sfpi::SFPEXMAN_MOD1_PAD9);  // 11
    TTI_SFPCAST(5, 5, 0);                    // 12
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG4,
        0);  // 13: P2 head
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG7,
        sfpi::SFPLOADI_MOD0_LOWER,
        (kNegativePositiveScaleBits & 0xffffu));  // 14: Horner gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        0);  // 15: P2 tail
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                 0x4380);       // 16: 2*scale == 256, also Horner gap
    TTI_SFPSETEXP(0, 4, 0, 2);  // 17: scaled exp(x) in L0
    TTI_SFPNOP;                 // 18: SETEXP result retirement
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        0);  // 19: s^2
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG3,
        0);  // 20: x*s, also square retirement gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG5,
        0);  // 21: s^2 + 256*s
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);                                          // 22: negative numerator, cross-term retirement gap
    TTI_SFPADDI(0x4700, ckernel::p_sfpu::LREG5, 0);  // 23: +32768
    TTI_SFPNOP;                                      // 24: denominator retirement
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, 0);  // 25
    TTI_SFPNOP;                                                           // 26: reciprocal seed retirement
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG5,
        2);                                         // 27: reciprocal Newton residual
    TTI_SFPNOP;                                     // 28
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);  // 29
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);                    // 30: refined reciprocal
    TTI_SFPENCC(3, 0, 0, 10);  // 31
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        0);  // 32: negative numerator / denominator

    auto same_row_suffix = [&]() {
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG5,
            0);  // positive factor, also negative-ratio retirement gap
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0);  // negative result, also positive-factor retirement gap
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG5,
            0);                                         // positive result
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0);  // x < 0
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        if constexpr (Config::kRawNegativeNanWord != 0) {
            // Physical BF16 sign/exponent and mantissa are disjoint. After
            // XOR, the second predicate excludes exact -Inf without a new mask.
            TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
            TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
            TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, Config::kRawNegativeNanWord);
            TTI_SFPENCC(0, 0, 0, 0);
        }
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG5,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, Advance, 0);
        restore_cross_row_pins();
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int row = 1; row < 32; ++row) {
        TTI_REPLAY(0, 32, 0, 0);
        same_row_suffix();
    }
}
}  // namespace sfpi
