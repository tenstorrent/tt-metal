// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace sfpi {
#if defined(ARCH_BLACKHOLE)
template <class Config, unsigned Hold, unsigned Advance, class Enter>
inline void factored_cw_bh(Enter enter) {
    constexpr uint32_t kRoundingBias = Config::kRoundingBiasBits;
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOADI_MOD0_UPPER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[4])) >> 16);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        sfpi::SFPLOADI_MOD0_LOWER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[4])) & 0xffffu);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_UPPER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[3])) >> 16);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG5,
        sfpi::SFPLOADI_MOD0_LOWER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[3])) & 0xffffu);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_UPPER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[2])) >> 16);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG6,
        sfpi::SFPLOADI_MOD0_LOWER,
        (__builtin_bit_cast(uint32_t, Config::kCoefficients[2])) & 0xffffu);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kRoundingBias) >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kRoundingBias) & 0xffffu);

    enter();
    TTI_REPLAY(0, 30, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kLowerBf16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 9);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kUpperBf16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPADD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    // Keep L7's rounding bias live across all 31 replays.  The selected leaf's
    // exact c1=0.5 is already preloaded in L14, so it is also the scale/bias
    // bit addend.  Overwriting L7 here made only recorded row zero correct;
    // every replay then rounded with 0.5 instead of 12582912.
    // The shift reads the preserved integer copy in L3, not the MAD-family
    // result in L1.  It is therefore the required independent issue between
    // the L1 producer and its later residual-FMA consumer; no NOP belongs here.
    TTI_SFPSHFT(0x017, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 7);
    TTI_SFPIADD(
        0,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG3,
        sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    // The reduced result is already the k==0 default output in L0.  The base
    // calculation below is independent and supplies its MAD hazard gap; the
    // reconstruction consumes L0 directly, so no L2->L0 copy is required.
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        Hold,
        0);  // raw BH DST word; fills the final MAD hazard slot
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, Config::kRawEqualValue);
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG2, 1);

    auto same_row_suffix = []() {
        static_assert(
            Config::kRawEqualMask == Config::kRawEqualValue &&
                (Config::kRawEqualMask & Config::kRawNonzeroMask) == 0u &&
                (Config::kRawEqualMask | Config::kRawNonzeroMask) == 0xffffu && Config::kRawCount > 0u,
            "verified replay raw predicate descriptor is inconsistent");
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPSETEXP(
            Config::kInfinityExponent,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG0,
            1);  // exact +Inf override value
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (uint32_t row = 1; row < 32; ++row) {
        TTI_REPLAY(0, 30, 0, 0);
        same_row_suffix();
    }
}
#elif defined(ARCH_WORMHOLE)
template <class Config, unsigned Hold, unsigned Advance, class Enter>
inline void factored_cw_wh(Enter enter) {
    // The typed gate proves BF16 identity with the ordinary WH graph over all inputs.
    // L4..L7 stay pinned; L12..L14 own the three programmable constants.
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[4])) >> 16);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[4])) & 0xffffu);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[3])) >> 16);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[3])) & 0xffffu);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[2])) >> 16);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(uint32_t, Config::kCoefficients[2])) & 0xffffu);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_UPPER, (0x4b400000u) >> 16);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_LOWER, (0x4b400000u) & 0xffffu);
    enter();
    TTI_REPLAY(0, 27, 1, 1);
    TTI_SFPLOAD(0, 0, Hold, 0);
    TTI_SFPLOADI(2, 0, Config::kLowerBf16);
    TTI_SFPSWAP(0, 2, 0, 9);
    TTI_SFPNOP;
    TTI_SFPLOADI(2, 0, Config::kUpperBf16);
    TTI_SFPSWAP(0, 2, 0, 1);
    TTI_SFPNOP;
    TTI_SFPMAD(0, 12, 7, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, 1, 3, 0);
    TTI_SFPMAD(7, 11, 1, 1, 0);
    TTI_SFPSHFT(23, 3, 3, 1);
    TTI_SFPIADD(0, 14, 3, 4);
    TTI_SFPMAD(1, 13, 0, 0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(0, 0, 9, 1, 0);
    TTI_SFPMAD(4, 0, 5, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 0, 6, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 0, 14, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 1, 0, 0, 0);
    TTI_SFPMAD(14, 11, 3, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(3, 0, 1, 2, 0);
    TTI_SFPSETCC(0, 1, 0, 2);
    auto suffix = []() {
        TTI_SFPADD(10, 2, 2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, 0, 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
        TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
        TTI_SFPXOR(0, 2, 3, 0);
        TTI_SFPAND(0, 3, 2, 0);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETEXP(255, 10, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, 0, Advance, 0);
    };
    suffix();
#pragma GCC unroll 32
    for (uint32_t row = 1; row < 32; ++row) {
        TTI_REPLAY(0, 27, 0, 0);
        suffix();
    }
}
#endif
}  // namespace sfpi
