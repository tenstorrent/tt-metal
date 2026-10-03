// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16 {
constexpr uint32_t kLog2Hold = ADDR_MOD_3;
constexpr uint32_t kLog2Advance = ADDR_MOD_2;

template <typename Config>
inline void init_log2() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kScaleBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kCoefficientBits[Config::kDegree - 1]);
    // The shared whole-tile callback advances one vector per store.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

template <typename Config, int Iterations = 8>
inline void calculate_log2() {
    static_assert(Iterations > 0);
    // Pin once for the callback's whole tile, as in the selected replay body.
    if constexpr (Config::kDegree >= 3u) {
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 2] >> 16);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 2] & 0xffffu);
    }
    if constexpr (Config::kDegree >= 4u) {
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 3] >> 16);
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 3] & 0xffffu);
    }
    if constexpr (Config::kDegree >= 5u) {
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 4] >> 16);
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 4] & 0xffffu);
    }
    if constexpr (Config::kDegree >= 6u) {
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 5] >> 16);
        TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 5] & 0xffffu);
    }
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, kLog2Hold, 0);  // x
    // Eleven-slot prelude: physical U16 exponent bits0..7 ignore sign,
    // so both signed zeros/subnormals select -Inf. The float sign test
    // and explicit nonzero exponent exclude zero inputs.
    // Intersecting with exponent<255 opens only positive finite normals.
    // L0..L3 are scratch; L4..L7 retain coefficient pins.
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_UINT16, kLog2Hold, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x00ff);
    // WH only supports the destructive two-register AND (mod1=0).
    // The mask load also supplies the raw LOAD's required gap.
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0xff80);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(0xF01, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPSETEXP(127, p_sfpu::LREG2, p_sfpu::LREG2, 1);  // m (imm exponent)
    TTI_SFPIADD(0xF81, p_sfpu::LREG1, p_sfpu::LREG1, 5);  // e -= 127 (imm, no CC)
    // Operand order mirrors the sfpi-emitted word exactly
    // (a=LCONST_1, b=m): 1.0*m + (-1.0) — bit-identical either way.
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        p_sfpu::LREG2,
        p_sfpu::LCONST_neg1,
        p_sfpu::LREG2,
        0);                                          // u = m + (-1.0) (hw const L11; MAD-unit producer)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // |e| (int); u-gap filler
    TTI_SFPMAD(p_sfpu::LREG13, p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG3,
               0);                                 // h = c[D]*u + c[D-1]
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);  // float(|e|); h-gap filler
    TTI_SFPMAD(
        p_sfpu::LREG3,
        p_sfpu::LREG2,
        (Config::kDegree >= 3) ? p_sfpu::LREG7 : p_sfpu::LCONST_0,
        p_sfpu::LREG3,
        0);                                             // rung 1: + c[D-2] (or c0 == 0 -> hw zero at DEG 2)
    TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // e_f={sgn e,mag |e|}
    if constexpr (Config::kDegree >= 3) {
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 4) ? p_sfpu::LREG6 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 2 (gapped by the SETSGN above)
    }
    if constexpr (Config::kDegree >= 4) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 5) ? p_sfpu::LREG5 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 3
    }
    if constexpr (Config::kDegree >= 5) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            (Config::kDegree >= 6) ? p_sfpu::LREG4 : p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 4
    }
    if constexpr (Config::kDegree >= 6) {
        TTI_SFPNOP;
        TTI_SFPMAD(
            p_sfpu::LREG3,
            p_sfpu::LREG2,
            p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // rung 5: + c0 == 0 (hw zero, gate-proved)
    }
    TTI_SFPNOP;
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        p_sfpu::LREG1,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        0);  // e_float + h (a=LCONST_1 form, mirrors the sfpi sfpadd)
    if constexpr (Config::kScaleBits != 0x3f800000u) {
        TTI_SFPNOP;
        TTI_SFPMUL(
            p_sfpu::LREG3,
            p_sfpu::LREG12,
            p_sfpu::LCONST_0,
            p_sfpu::LREG3,
            0);  // * LOG_HW_SCALE (prgm0; exact-0.0 addend == sfpmul)
    }
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);        // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kLog2Advance, 0);  // dest += 2
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        if constexpr (Config::kTailSlots != 0u) {
            TTI_SFPSTORE(p_sfpu::LREG3, 0, kLog2Advance, 0);
        }
    }
}
}  // namespace ckernel::sfpu::bf16
