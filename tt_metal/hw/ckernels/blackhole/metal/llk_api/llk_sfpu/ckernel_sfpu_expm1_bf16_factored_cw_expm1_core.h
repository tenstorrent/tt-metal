// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_expm1_bf16_replay_window.h"
namespace ckernel::sfpu::ttpoly::expm1 {
using namespace ::sfpi;
using ::sfpi::DataLayout;
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
    auto row_head = [&] {
        TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
        // x * 1 + 0 turns either NaN into +NaN, which the upper clamp sends to +Inf.
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kLowerBf16);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 9);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kUpperBf16);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);
        TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
        TTI_SFPNOP;
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
        // L7 keeps the rounding bias for every replay; c1 = 0.5 in L14 is the bias addend.
        TTI_SFPSHFT(0x017, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 7);
        TTI_SFPIADD(
            0,
            ckernel::p_sfpu::LREG14,
            ckernel::p_sfpu::LREG3,
            sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
        TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    };
    auto row_tail = [&] {
        TTI_SFPNOP;
        TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
        // L0 already holds the k == 0 result, which the reconstruction reads.
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);
        TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
    };
    ::ckernel::sfpu::bf16::expm1::record_split<0, 27, 16>(row_head, row_tail);
    auto same_row_suffix = []() {
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
        // A store through the hold mode leaves the DEST counter for one row step.
        if constexpr (Advance == Hold) {
            TTI_INCRWC(0, 2, 0, 0);
        }
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (uint32_t row = 1; row < 32; ++row) {
        ::ckernel::sfpu::bf16::expm1::replay_split<0, 27, 16>(row_head, row_tail);
        same_row_suffix();
    }
}
}  // namespace ckernel::sfpu::ttpoly::expm1
