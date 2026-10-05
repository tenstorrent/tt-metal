// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
// Include inside namespace sfpi after sfpi_min_max.h; callers own traversal/init.

template <uint32_t A, uint32_t B, uint32_t Hold, uint32_t Advance>
inline void simple_finish_pair() {
    ::ckernel::sfpu::bf16_sfpi::sfp_stoch_rnd(SFPSTOCHRND_RND_EVEN, 0, A, A, A, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    ::ckernel::sfpu::bf16_sfpi::sfp_stoch_rnd(SFPSTOCHRND_RND_EVEN, 0, B, B, B, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    ::ckernel::sfpu::bf16_sfpi::sfpstore(A, 0, Hold, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpstore(B, 0, Advance, 2);  // pair complete, dest += 4
}

template <
    uint32_t A,
    uint32_t B,
    uint32_t RAWA,
    uint32_t RAWB,
    uint32_t MASK,
    uint32_t SCRATCH,
    uint32_t EqualValue,
    uint32_t NonzeroMask,
    uint32_t Output,
    uint32_t Hold>
inline void simple_raw_terminal_pair() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(RAWA, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpload(RAWB, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 2);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(MASK, sfpi::SFPLOADI_MOD0_USHORT, EqualValue);
    ::ckernel::sfpu::bf16_sfpi::sfpxor(0, MASK, RAWA, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpxor(0, MASK, RAWB, 0);
    if constexpr (NonzeroMask == 0u) {
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfploadi(A, sfpi::SFPLOADI_MOD0_FLOATB, Output, cc_region);
            cc_region.close();
        };
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfploadi(B, sfpi::SFPLOADI_MOD0_FLOATB, Output, cc_region);
            cc_region.close();
        };
    } else {
        ::ckernel::sfpu::bf16_sfpi::sfpand(MASK, RAWA, SCRATCH, 1);
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_NE0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfploadi(A, sfpi::SFPLOADI_MOD0_FLOATB, Output, cc_region);
            cc_region.close();
        };
        ::ckernel::sfpu::bf16_sfpi::sfpand(MASK, RAWB, SCRATCH, 1);
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_NE0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfploadi(B, sfpi::SFPLOADI_MOD0_FLOATB, Output, cc_region);
            cc_region.close();
        };
    }
}

template <
    bool OriginHalfTie,
    bool RawTerminal,
    uint32_t EqualValue,
    uint32_t NonzeroMask,
    uint32_t Output,
    uint32_t Hold,
    uint32_t Advance>
inline void simple_gated_pair() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // xA
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG1, 0, Hold, 2);  // xB
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
    ::ckernel::sfpu::bf16_sfpi::sfpmul(
        ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpmul(
        ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 0);
    if constexpr (OriginHalfTie) {
        ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);     // |xA|
        ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);     // |xB|
        ::ckernel::sfpu::bf16_sfpi::sfpdivp2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // *2^126
        ::ckernel::sfpu::bf16_sfpi::sfpdivp2(0x07E, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 1);  // *2^126
        ::ckernel::sfpu::bf16_sfpi::sfpaddi(0xBFFF, ckernel::p_sfpu::LREG4, 0);  // scaled |xA| - 1.9921875
        ::ckernel::sfpu::bf16_sfpi::sfpaddi(0xBFFF, ckernel::p_sfpu::LREG5, 0);  // scaled |xB| - 1.9921875
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpsetman(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1, cc_region);
            cc_region.close();
        };
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpsetman(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1, cc_region);
            cc_region.close();
        };
    }
    if constexpr (RawTerminal) {
        simple_raw_terminal_pair<
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            EqualValue,
            NonzeroMask,
            Output,
            Hold>();
    }
    simple_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, Hold, Advance>();
}
