// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel_sfpu_sqrt_bf16_newton_root.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_sqrt() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
// The kernel's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_sqrt_bf16_scheduled() {
    TTI_SFPLOADI(7, 2, 0x8000);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPSHFT(0xfff, 0, 0, 1);
        TTI_SFPIADD(0, 12, 0, 6);
        TTI_SFPMUL(2, 0, 9, 1, 0);
        TTI_SFPMOV(0, 0, 3, 1);
        TTI_SFPMUL(3, 1, 9, 1, 0);
        TTI_SFPLOAD(3, 0, 3, 2);
        TTI_SFPADD(10, 14, 1, 4, 0);
        TTI_SFPABS(0, 2, 2, 1);
        TTI_SFPMAD(1, 4, 13, 1, 0);
        TTI_SFPLOADI(6, 0, 0x7f80);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(2, 0, 9, 1, 0);
        TTI_SFPMOV(0, 0, 4, 1);
        TTI_SFPMOV(0, 3, 5, 0);
        TTI_SFPSHFT(0xfff, 5, 5, 1);
        TTI_SFPIADD(0, 12, 5, 6);
        TTI_SFPMAD(4, 1, 10, 4, 0);
        TTI_SFPIADD(0, 2, 6, 2);
        TTI_SFPDIVP2(0xfff, 4, 2, 1);
        TTI_SFPMAD(2, 1, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(3, 5, 9, 1, 0);
        TTI_SFPMOV(0, 5, 2, 1);
        TTI_SFPMUL(2, 1, 9, 1, 0);
        TTI_SFPLOAD(2, 6, 3, 0);
        TTI_SFPADD(10, 14, 1, 4, 0);
        TTI_SFPABS(0, 3, 3, 1);
        TTI_SFPMAD(1, 4, 13, 1, 0);
        TTI_SFPIADD(0, 7, 2, 2);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(5, 1, 9, 1, 0);
        TTI_SFPLOADI(6, 0, 0x7f80);
        TTI_SFPMUL(3, 1, 9, 2, 0);
        TTI_SFPMOV(0, 1, 4, 1);
        TTI_SFPMAD(4, 2, 10, 4, 0);
        TTI_SFPIADD(0, 3, 6, 2);
        TTI_SFPDIVP2(0xfff, 4, 3, 1);
        TTI_SFPMAD(3, 2, 2, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(2, 6, 3, 2);
        TTI_SFPIADD(0, 7, 2, 2);
        TTI_SFPLOADI(1, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPLOAD(2, 0, 3, 4);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPSHFT(0xfff, 0, 0, 1);
        TTI_SFPIADD(0, 12, 0, 6);
        TTI_SFPMUL(2, 0, 9, 3, 0);
        TTI_SFPMOV(0, 0, 4, 1);
        TTI_SFPMUL(4, 3, 9, 3, 0);
        TTI_SFPABS(0, 2, 2, 1);
        TTI_SFPADD(10, 14, 3, 4, 0);
        TTI_SFPLOADI(6, 0, 0x7f80);
        TTI_SFPMAD(3, 4, 13, 3, 0);
        TTI_SFPLOAD(4, 6, 3, 4);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPSTORE(1, 0, 3, 2);
        TTI_SFPLOAD(1, 0, 3, 6);
        TTI_SFPMUL(2, 0, 9, 3, 0);
        TTI_SFPMOV(0, 0, 5, 1);
        TTI_SFPMAD(5, 3, 10, 5, 0);
        TTI_SFPIADD(0, 2, 6, 2);
        TTI_SFPDIVP2(0xfff, 5, 2, 1);
        TTI_SFPMAD(2, 3, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMOV(0, 1, 2, 0);
        TTI_SFPSHFT(0xfff, 2, 2, 1);
        TTI_SFPIADD(0, 12, 2, 6);
        TTI_SFPMUL(1, 2, 9, 3, 0);
        TTI_SFPMOV(0, 2, 5, 1);
        TTI_SFPMUL(5, 3, 9, 3, 0);
        TTI_SFPABS(0, 1, 1, 1);
        TTI_SFPADD(10, 14, 3, 5, 0);
        TTI_SFPIADD(0, 7, 4, 2);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(3, 5, 13, 3, 0);
        TTI_SFPLOADI(6, 0, 0x7f80);
        TTI_SFPMUL(2, 3, 9, 2, 0);
        TTI_SFPLOAD(3, 6, 3, 6);
        TTI_SFPMUL(1, 2, 9, 4, 0);
        TTI_SFPMOV(0, 2, 5, 1);
        TTI_SFPMAD(5, 4, 10, 5, 0);
        TTI_SFPIADD(0, 1, 6, 2);
        TTI_SFPDIVP2(0xfff, 5, 1, 1);
        TTI_SFPMAD(1, 4, 4, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0, 7, 3, 2);
        TTI_SFPLOADI(2, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 1, 1);
        TTI_SFPSTORE(0, 0, 3, 4);
        TTI_SFPSTORE(1, 0, 3, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_sqrt_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_sqrt_bf16_scheduled();
}

}  // namespace ckernel::sfpu
