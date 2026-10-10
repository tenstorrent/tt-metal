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

template <uint32_t Hold, uint32_t Advance, bool FoldIdentityStore = false>
inline void simple_threshold_pair() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG1, 0, Hold, 2);
    ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
    ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
    if constexpr (FoldIdentityStore) {
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfple(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 1, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpstore(ckernel::p_sfpu::LCONST_0, 0, Hold, 0, cc_region);
            cc_region.close();
        };
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfple(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 1, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpstore(ckernel::p_sfpu::LCONST_0, 0, Hold, 2, cc_region);
            cc_region.close();
        };
        ::ckernel::sfpu::bf16_sfpi::incrwc(0, 4, 0, 0);
    } else {
        ::ckernel::sfpu::bf16_sfpi::sfpmad(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, 0);
        ::ckernel::sfpu::bf16_sfpi::sfpmad(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, ckernel::p_sfpu::LREG4, 0, SFPSETCC_MOD1_LREG_LT0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpmov(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0, cc_region);
            cc_region.close();
        };
        {
            ::ckernel::sfpu::bf16_sfpi::Region cc_region;
            ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, ckernel::p_sfpu::LREG5, 0, SFPSETCC_MOD1_LREG_LT0, cc_region);
            ::ckernel::sfpu::bf16_sfpi::sfpmov(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0, cc_region);
            cc_region.close();
        };
        simple_finish_pair<ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, Hold, Advance>();
    }
}

template <uint32_t EqualValue, uint32_t Hold, uint32_t Advance>
inline void simple_bh_gated_joint_row() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // 0: x
    ::ckernel::sfpu::bf16_sfpi::sfpload(
        ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);  // 1: raw U16 / x-load gap
    ::ckernel::sfpu::bf16_sfpi::sfpmad(
        ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);  // 2: gate
    ::ckernel::sfpu::bf16_sfpi::sfploadi(
        ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT, EqualValue);  // 3: gate gap / raw anchor
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);    // 4
    ::ckernel::sfpu::bf16_sfpi::sfpxor(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);        // 5
    ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);     // 6
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);    // 7
    ::ckernel::sfpu::bf16_sfpi::sfpdivp2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // 8
    // Raw partitioning is independent of L4 and fills the required
    // DIVP2->ADDI gap.
    ::ckernel::sfpu::bf16_sfpi::sfpand(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        1);                                                                  // 9: negative exponent-FF partition delta
    ::ckernel::sfpu::bf16_sfpi::sfpaddi(0xBFFF, ckernel::p_sfpu::LREG4, 0);  // 10
    // The independent x*gate product fills the ADDI->SETCC gap.  It is also
    // separated from its gate-clamp producer at slot 7 by three issues.  No
    // CC window is active across either filler.
    ::ckernel::sfpu::bf16_sfpi::sfpmul(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 11: x*gate
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(
            0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);                              // 12
        ::ckernel::sfpu::bf16_sfpi::sfpsetman(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1, cc_region);  // 13
        cc_region.close();
    };  // 14
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(
            0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);  // 15
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(
            0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0, cc_region);  // 16
        ::ckernel::sfpu::bf16_sfpi::sfploadi(
            ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0xFF80, cc_region);  // 17
        cc_region.close();
    };  // 18
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(
            0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0, cc_region);  // 19
        ::ckernel::sfpu::bf16_sfpi::sfploadi(
            ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x0000, cc_region);  // 20
        cc_region.close();
    };  // 21
    ::ckernel::sfpu::bf16_sfpi::sfp_stoch_rnd(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);                                    // 22
    ::ckernel::sfpu::bf16_sfpi::sfpnop();                                         // 23: RNE/store dependency
    ::ckernel::sfpu::bf16_sfpi::sfpstore(ckernel::p_sfpu::LREG2, 0, Advance, 0);  // 24
}

sfpi_inline vFloat simple_threshold_identity(vFloat x, float lambda) {
    vFloat y = x;
    vFloat ax = setsgn(x, 0);
    v_if(ax <= lambda) { y = 0.0f; }
    v_endif;

    return y;
}

template <bool NativeOrderedBranches = false>
sfpi_inline vFloat simple_threshold_softshift(vFloat x, float lambda) {
    if constexpr (NativeOrderedBranches) {
        vFloat y = 0.0f;
        v_if(x > lambda) { y = x - lambda; }
        v_elseif(x < -lambda) { y = x + lambda; }
        v_endif;
        return y;
    } else {
        vFloat ax = setsgn(x, 0);
        vFloat y = 0.0f;
        v_if(ax > lambda) {
            y = ax - lambda;
            v_if(x < 0.0f) { y = -y; }
            v_endif;
        }
        v_endif;
        return y;
    }
}

sfpi_inline vFloat simple_gated_product(vFloat x, float q0, float q1) {
    vFloat gate = q1 * x + q0;
    vFloat zero = 0.0f;
    vFloat one = 1.0f;
    ordered_min_max(zero, gate);  // gate = max(0, gate)
    ordered_min_max(gate, one);   // gate = min(gate, 1)
    vFloat y = x * gate;
    return y;
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
