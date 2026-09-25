// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi after sfpi_min_max.h; callers own traversal/init.

template <uint32_t A, uint32_t B, uint32_t Hold, uint32_t Advance>
inline void simple_finish_pair() {
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, A, A, A, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, B, B, B, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(A, 0, Hold, 0);
    TTI_SFPSTORE(B, 0, Advance, 2);  // pair complete, dest += 4
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
    TTI_SFPLOAD(RAWA, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    TTI_SFPLOAD(RAWB, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 2);
    TTI_SFPLOADI(MASK, sfpi::SFPLOADI_MOD0_USHORT, EqualValue);
    TTI_SFPXOR(0, MASK, RAWA, 0);
    TTI_SFPXOR(0, MASK, RAWB, 0);
    if constexpr (NonzeroMask == 0u) {
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(A, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(B, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
#if defined(ARCH_WORMHOLE)
        TTI_SFPMOV(0, RAWA, SCRATCH, 0);
        TTI_SFPAND(0, MASK, SCRATCH, 0);
#else
        TTI_SFPAND(MASK, RAWA, SCRATCH, 1);
#endif
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(A, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
#if defined(ARCH_WORMHOLE)
        TTI_SFPMOV(0, RAWB, SCRATCH, 0);
        TTI_SFPAND(0, MASK, SCRATCH, 0);
#else
        TTI_SFPAND(MASK, RAWB, SCRATCH, 1);
#endif
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(B, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <uint32_t Hold, uint32_t Advance>
inline void simple_threshold_pair() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 2);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    simple_finish_pair<ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, Hold, Advance>();
}

template <uint32_t Output, uint32_t Hold, uint32_t Advance>
inline void simple_wh_gated_row() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);                              // 0: x
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);  // 1: raw / x-load gap
    // The WH SFPI baseline emits MUL then ADDI, not a contracted MAD.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);                                                                  // 2: q1*x
    TTI_SFPAND(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, 0);        // 3: raw mask / MUL gap
    TTI_SFPADDI(0x3F00, ckernel::p_sfpu::LREG2, 0);                          // 4: +q0, separate rounding
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, 0);        // 5: terminal delta / ADD gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);    // 6: max(0, gate)
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);     // 7: abs(x)
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);    // 8: min(gate, 1)
    TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // 9: scale abs(x)
    TTI_SFPNOP;                                                              // 10: DIVP2 -> ADDI gap
    TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);                          // 11: boundary delta
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);                                                                    // 12: x * gate
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 13
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);       // 14: signed MIN_NORMAL
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 15
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Output);  // 17
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 18
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);            // 19
    TTI_SFPNOP;                                           // 20: round/store gap
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0);  // 21
}

template <uint32_t EqualValue, uint32_t Hold, uint32_t Advance>
inline void simple_bh_gated_joint_row() {
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // 0: x
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold,
                0);  // 1: raw U16 / x-load gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);  // 2: gate
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT,
                 EqualValue);                                                // 3: gate gap / raw anchor
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);    // 4
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);        // 5
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);     // 6
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);    // 7
    TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // 8
    // Raw partitioning is independent of L4 and fills the required
    // DIVP2->ADDI gap.
    TTI_SFPAND(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        1);                                          // 9: negative exponent-FF partition delta
    TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);  // 10
    // The independent x*gate product fills the ADDI->SETCC gap.  It is also
    // separated from its gate-clamp producer at slot 7 by three issues.  No
    // CC window is active across either filler.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);                                                                    // 11: x*gate
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 12
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);       // 13
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 14
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 15
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0xFF80);  // 17
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 18
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 19
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x0000);  // 20
    TTI_SFPENCC(0, 0, 0, 0);                                                   // 21
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);            // 22
    TTI_SFPNOP;                                           // 23: RNE/store dependency
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0);  // 24
}

sfpi_inline vFloat simple_threshold_identity(vFloat x, float lambda) {
    vFloat y = x;
    vFloat ax = setsgn(x, 0);
    v_if(ax <= lambda) { y = 0.0f; }
    v_endif;

    return y;
}

sfpi_inline vFloat simple_threshold_softshift(vFloat x, float lambda) {
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
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 2);  // xB
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 0);
    if constexpr (OriginHalfTie) {
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);     // |xA|
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);     // |xB|
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);  // *2^126
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 1);  // *2^126
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);                          // scaled |xA| - 1.9921875
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);                          // scaled |xB| - 1.9921875
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
        TTI_SFPENCC(0, 0, 0, 0);
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
