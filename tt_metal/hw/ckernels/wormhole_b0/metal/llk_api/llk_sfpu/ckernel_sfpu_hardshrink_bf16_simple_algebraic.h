// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::hardshrink {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <uint32_t Hold, uint32_t Advance>
inline void simple_threshold_pair() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 2);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, 4);
    TTI_SFPSTORE(ckernel::p_sfpu::LCONST_0, 0, Hold, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, 4);
    TTI_SFPSTORE(ckernel::p_sfpu::LCONST_0, 0, Hold, 2);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_INCRWC(0, 4, 0, 0);
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
