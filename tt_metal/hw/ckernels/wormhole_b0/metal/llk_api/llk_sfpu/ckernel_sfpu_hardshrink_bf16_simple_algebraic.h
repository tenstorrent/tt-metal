// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// StepEach: Advance steps one row (dest += 2, stock's ADDR_MOD_6), so both stores advance.
namespace ckernel::sfpu::ttpoly::hardshrink {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <uint32_t A, uint32_t B, uint32_t Hold, uint32_t Advance, bool StepEach = false>
inline void simple_finish_pair() {
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, A, A, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, B, B, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    if constexpr (StepEach) {
        TTI_SFPSTORE(A, 0, Advance, 0);
        TTI_SFPSTORE(B, 0, Advance, 0);
    } else {
        TTI_SFPSTORE(A, 0, Hold, 0);
        TTI_SFPSTORE(B, 0, Advance, 2);  // pair complete, dest += 4
    }
}

template <uint32_t Hold, uint32_t Advance, bool FoldIdentityStore = false, bool StepEach = false>
inline void simple_threshold_pair() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 2);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
    if constexpr (FoldIdentityStore) {
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 4);
        TTI_SFPSTORE(ckernel::p_sfpu::LCONST_0, 0, Hold, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, 4);
        TTI_SFPSTORE(ckernel::p_sfpu::LCONST_0, 0, Hold, 2);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_INCRWC(0, 4, 0, 0);
    } else {
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, 0);
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        simple_finish_pair<ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, Hold, Advance, StepEach>();
    }
}

template <uint32_t Output, uint32_t Hold, uint32_t Advance>
inline void simple_wh_gated_row_tail() {
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Output);  // 17
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 18
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);            // 19
    TTI_SFPNOP;                                           // 20: round/store gap
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0);  // 21
}
}  // namespace ckernel::sfpu::ttpoly::hardshrink
