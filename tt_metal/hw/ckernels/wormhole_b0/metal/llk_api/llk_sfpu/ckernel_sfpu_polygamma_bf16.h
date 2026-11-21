// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_polygamma() {
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
inline void calculate_polygamma_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 32; ++pass) {
        TTI_SFPLOAD(1, 0, 3, 0);
        TTI_SFPMOV(0, 1, 0, 2);
        TTI_SFPSETSGN(0, 1, 3, 1);
        TTI_SFPSETSGN(0, 10, 0, 0);
        TTI_SFPADD(10, 3, 10, 3, 0);
        TTI_SFPMOV(0, 1, 2, 2);
        TTI_SFPMULI(0x3f00, 0, 0);
        TTI_SFPADDI(0x4b40, 2, 0);
        TTI_SFPMOV(0, 0, 0, 1);
        TTI_SFPMOV(0, 3, 6, 2);
        TTI_SFPADDI(0xcb40, 2, 0);
        TTI_SFPADDI(0x3f00, 0, 0);
        TTI_SFPSETMAN(0, 11, 6, 0);
        TTI_SFPMAD(12, 6, 13, 4, 0);
        TTI_SFPMUL(0, 2, 9, 2, 0);
        TTI_SFPMAD(4, 6, 14, 4, 0);
        TTI_SFPMAD(2, 11, 1, 2, 0);
        TTI_SFPNOT(0, 3, 5, 0);
        TTI_SFPMAD(6, 4, 10, 6, 0);
        TTI_SFPSETMAN(0, 5, 5, 1);
        TTI_SFPMOV(0, 2, 7, 2);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPMAD(4, 6, 4, 4, 0);
        TTI_SFPSETMAN(0, 11, 7, 0);
        TTI_SFPMAD(12, 7, 13, 6, 0);
        TTI_SFPMUL(4, 5, 9, 4, 0);
        TTI_SFPMAD(6, 7, 14, 5, 0);
        TTI_SFPSETSGN(0, 4, 3, 0);
        TTI_SFPLOADI(4, 0, 0x3e21);
        TTI_SFPLOADI(6, 0, 0x3f00);
        TTI_SFPMAD(3, 4, 6, 4, 0);
        TTI_SFPNOT(0, 2, 6, 0);
        TTI_SFPMAD(7, 5, 10, 7, 0);
        TTI_SFPSETMAN(0, 6, 6, 1);
        TTI_SFPMAD(4, 3, 10, 4, 0);
        TTI_SFPMULI(0x3f00, 6, 0);
        TTI_SFPMAD(5, 7, 5, 5, 0);
        TTI_SFPMUL(0, 2, 9, 7, 0);
        TTI_SFPMUL(4, 3, 9, 4, 0);
        TTI_SFPMUL(7, 7, 9, 3, 0);
        TTI_SFPMUL(5, 6, 9, 5, 0);
        TTI_SFPLOADI(6, 0, 0x4193);
        TTI_SFPLOADI(7, 0, 0x40b4);
        TTI_SFPMAD(3, 6, 7, 6, 0);
        TTI_SFPMOV(0, 2, 7, 2);
        TTI_SFPSETSGN(0, 5, 7, 0);
        TTI_SFPMOV(0, 1, 5, 2);
        TTI_SFPSETSGN(0, 4, 5, 0);
        TTI_SFPLOADI(4, 0, 0x4053);
        TTI_SFPMAD(6, 3, 4, 3, 0);
        TTI_SFPMAD(7, 7, 5, 5, 0);
        TTI_SFPSETSGN(0, 1, 4, 1);
        TTI_SFPMAD(0, 3, 5, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPEXEXP(0, 4, 2, 1);
        TTI_SFPIADD(0xf01, 2, 2, 5);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPMOV(0, 1, 2, 2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSHFT(1, 0, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(2, 0, 0x7e80);
        TTI_SFPIADD(0, 1, 2, 6);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(0, 0, 0x80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPEXEXP(0, 2, 2, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPIADD(0xf01, 2, 2, 5);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPIADD(0, 1, 2, 6);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(2, 0, 0xff80);
        TTI_SFPIADD(0, 1, 2, 6);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_polygamma_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_polygamma_bf16_scheduled();
}

}  // namespace ckernel::sfpu
