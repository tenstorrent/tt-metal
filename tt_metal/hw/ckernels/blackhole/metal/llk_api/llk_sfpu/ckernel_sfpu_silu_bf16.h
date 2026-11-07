// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

// The fitted exp coefficients the body reads from Prgm1 (LREG13) and Prgm2 (LREG14), pre-scaled.
inline constexpr float kSiluBf16Prgm1 = __builtin_bit_cast(float, 0x3eaca410u) * 0x1p-46f;
inline constexpr float kSiluBf16Prgm2 = __builtin_bit_cast(float, 0x3f285adeu) * 0x1p-23f;

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_silu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}

// Generated BF16 silu over the whole tile, 8 passes of 4 rows; stores the same word for every BF16 input.
inline void calculate_silu_bf16_scheduled() {
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 0);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(4, 8, 0x3f80);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPMAD(13, 5, 14, 6, 0);
        TTI_SFPLOADI(4, 0xa, 0x3884);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPLOAD(0, 0, 7, 2);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPLOADI(6, 0, 0x430f);
        TTI_SFPDIVP2(0x7e, 2, 7, 1);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPADDI(0xbfff, 7, 0);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 7, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPMAD(1, 3, 6, 7, 0);
        TTI_SFPADD(10, 1, 0, 6, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFPEXEXP(0, 7, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 5, 1);
        TTI_SFPSTORE(5, 0, 7, 0);
        TTI_SFPEXMAN(0, 7, 5, 0);
        TTI_SFPSHFT(0, 2, 5, 0);
        TTI_SFPEXMAN(0, 5, 2, 1);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPAND(5, 7, 5, 1);
        TTI_SFPMAD(13, 2, 14, 7, 0);
        TTI_SFPMAD(7, 2, 4, 2, 0);
        TTI_SFPSETEXP(0, 2, 5, 2);
        TTI_SFPMUL(1, 5, 9, 2, 0);
        TTI_SFPMULI(0x3780, 5, 0);
        TTI_SFPMULI(0x3800, 2, 0);
        TTI_SFPADD(10, 5, 10, 5, 0);
        TTI_SFPARECIP(0, 5, 7, 0);
        TTI_SFPMAD(5, 7, 12, 5, 2);
        TTI_SFPMAD(5, 7, 9, 5, 3);
        TTI_SFPLOAD(7, 0, 7, 4);
        TTI_SFPMAD(2, 5, 6, 2, 1);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPDIVP2(0x7e, 1, 6, 1);
        TTI_SFPSWAP(0, 2, 0, 1);
        TTI_SFPADDI(0xbfff, 6, 0);
        TTI_SFPMULI(0x3f00, 2, 0);
        TTI_SFPSETCC(0, 6, 0, 6);
        TTI_SFPSETMAN(0, 0, 2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETSGN(0, 7, 0, 1);
        TTI_SFPMAD(0, 3, 5, 5, 0);
        TTI_SFPADD(10, 0, 7, 6, 0);
        TTI_SFPSWAP(0, 1, 2, 1);
        TTI_SFPEXEXP(0, 5, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 2);
        TTI_SFPEXMAN(0, 5, 2, 0);
        TTI_SFPSHFT(0, 1, 2, 0);
        TTI_SFPEXMAN(0, 2, 1, 1);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPGT(0, 9, 5, 8);
        TTI_SFPAND(2, 5, 2, 1);
        TTI_SFPMAD(13, 1, 14, 5, 0);
        TTI_SFPMAD(5, 1, 4, 5, 0);
        TTI_SFPSETEXP(0, 5, 2, 2);
        TTI_SFPMUL(0, 2, 9, 1, 0);
        TTI_SFPMULI(0x3780, 2, 0);
        TTI_SFPMULI(0x3800, 1, 0);
        TTI_SFPADD(10, 2, 10, 2, 0);
        TTI_SFPARECIP(0, 2, 5, 0);
        TTI_SFPMAD(2, 5, 12, 2, 2);
        TTI_SFPMAD(2, 5, 9, 5, 3);
        TTI_SFPLOAD(2, 0, 7, 6);
        TTI_SFPMAD(1, 5, 6, 1, 1);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 2, 6, 1);
        TTI_SFPMAD(6, 3, 5, 3, 0);
        TTI_SFPDIVP2(0x7e, 0, 5, 1);
        TTI_SFPSWAP(0, 1, 7, 1);
        TTI_SFPADDI(0xbfff, 5, 0);
        TTI_SFPMULI(0x3f00, 1, 0);
        TTI_SFPSETCC(0, 5, 0, 6);
        TTI_SFPSETMAN(0, 7, 1, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPEXEXP(0, 3, 5, 0);
        TTI_SFPEXMAN(0, 3, 7, 0);
        TTI_SFPSHFT(0, 5, 7, 0);
        TTI_SFPEXMAN(0, 7, 5, 1);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPGT(0, 9, 3, 8);
        TTI_SFPAND(7, 3, 3, 1);
        TTI_SFPMAD(13, 5, 14, 7, 0);
        TTI_SFPSWAP(0, 0, 1, 1);
        TTI_SFPMAD(7, 5, 4, 5, 0);
        TTI_SFPADD(10, 6, 2, 7, 0);
        TTI_SFPSETEXP(0, 5, 3, 2);
        TTI_SFPMUL(6, 3, 9, 0, 0);
        TTI_SFPMULI(0x3780, 3, 0);
        TTI_SFPMULI(0x3800, 0, 0);
        TTI_SFPADD(10, 3, 10, 3, 0);
        TTI_SFPDIVP2(0x7e, 6, 5, 1);
        TTI_SFPARECIP(0, 3, 4, 0);
        TTI_SFPMAD(3, 4, 12, 3, 2);
        TTI_SFPADDI(0xbfff, 5, 0);
        TTI_SFPMAD(3, 4, 9, 3, 3);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPMAD(0, 3, 7, 0, 1);
        TTI_SFPSTORE(1, 0, 7, 4);
        TTI_SFPSWAP(0, 0, 2, 1);
        TTI_SFPMULI(0x3f00, 0, 0);
        TTI_SFPSETCC(0, 5, 0, 6);
        TTI_SFPSETMAN(0, 2, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 6, 0, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

// Programs its own Prgm1/Prgm2 per call and puts back silu_init's Prgm1 = 1/ln2 for the stock path, so it
// depends on no init state beyond Prgm0 = 2.0f, which stock silu needs too.
inline void calculate_silu_bf16() {
    sfpi::vConstFloatPrgm1 = kSiluBf16Prgm1;
    sfpi::vConstFloatPrgm2 = kSiluBf16Prgm2;
    calculate_silu_bf16_scheduled();
    sfpi::vConstFloatPrgm1 = EXP_21F_ONE_LN2;
}

}  // namespace ckernel::sfpu
