// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Exact paired exponent-ALU body. Callers own replay, stores and traversal.
// L12/L13/L14 hold multiplier and the first two scaled coefficients;
// L5 holds 127 and L7 the final register addend. Clamp callers repin L2/L6
// to 255 before every issue because SWAP destroys those operands.
namespace sfpi {
template <uint32_t Degree, bool Clamp, uint32_t Hold>
inline void exp2_paired_body() {
    static_assert(Degree == 2u || Degree == 3u);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, Hold, 2);
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG3, 0);
    if constexpr (Clamp) {
        TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 9);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG3, 1);
    }
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPSHFT(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPCAST(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG6, 0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG7,
        (Degree == 3) ? ckernel::p_sfpu::LREG2 : ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG7,
        (Degree == 3) ? ckernel::p_sfpu::LREG6 : ckernel::p_sfpu::LREG4,
        0);
    if constexpr (Degree == 3) {
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG1, 0);
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG4, 0);
    }
    TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 2);
    TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 2);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
}
}  // namespace sfpi
