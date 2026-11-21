// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_sigmoid() {
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
inline void calculate_sigmoid_bf16_scheduled() {
    TTI_SFPLOADI(7, 0, 0x42fe);
    TTI_SFPLOADI(6, 2, 0xa410);
    TTI_SFPLOADI(6, 8, 0x27ac);
    TTI_SFPLOADI(5, 2, 0x3884);
    TTI_SFPLOADI(5, 8, 0x3f80);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOADI(4, 2, 0xaa3b);
        TTI_SFPLOADI(4, 8, 0xbfb8);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPMAD(0, 4, 7, 0, 0);
        TTI_SFPLOADI(1, 0, 0x437f);
        TTI_SFPSWAP(0, 0, 9, 1);
        TTI_SFPSWAP(0, 1, 0, 1);
        TTI_SFPEXEXP(0, 0, 1, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPSHFT(0, 1, 0, 0);
        TTI_SFPLOADI(2, 2, 0x5ade);
        TTI_SFPEXMAN(0, 0, 1, 1);
        TTI_SFPLOADI(2, 8, 0x33a8);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPMAD(6, 1, 2, 3, 0);
        TTI_SFPEXEXP(0, 0, 0, 1);
        TTI_SFPMAD(3, 1, 5, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETEXP(0, 1, 0, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPLOAD(1, 0, 3, 2);
        TTI_SFPMOV(0, 0, 3, 2);
        TTI_SFPSETMAN(0, 11, 3, 0);
        TTI_SFPMAD(1, 4, 7, 1, 0);
        TTI_SFPMAD(12, 3, 13, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(4, 3, 14, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(3, 4, 10, 3, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(4, 3, 4, 3, 0);
        TTI_SFPNOT(0, 0, 4, 0);
        TTI_SFPSETMAN(0, 4, 4, 1);
        TTI_SFPMULI(0x3f00, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(3, 4, 9, 3, 0);
        TTI_SFPLOADI(4, 0, 0x437f);
        TTI_SFPSWAP(0, 1, 9, 1);
        TTI_SFPSWAP(0, 4, 1, 1);
        TTI_SFPEXEXP(0, 1, 4, 0);
        TTI_SFPEXMAN(0, 1, 1, 0);
        TTI_SFPSHFT(0, 4, 1, 0);
        TTI_SFPEXMAN(0, 1, 4, 1);
        TTI_SFPCAST(4, 4, 0);
        TTI_SFPMAD(6, 4, 2, 2, 0);
        TTI_SFPEXEXP(0, 1, 1, 1);
        TTI_SFPMAD(2, 4, 5, 2, 0);
        TTI_SFPNOP;
        TTI_SFPSETEXP(0, 2, 1, 0);
        TTI_SFPADD(10, 1, 10, 1, 0);
        TTI_SFPSETSGN(0, 3, 0, 0);
        TTI_SFPMOV(0, 1, 2, 2);
        TTI_SFPSETMAN(0, 11, 2, 0);
        TTI_SFPMAD(12, 2, 13, 3, 0);
        TTI_SFPNOT(0, 1, 4, 0);
        TTI_SFPMAD(3, 2, 14, 3, 0);
        TTI_SFPSETMAN(0, 4, 4, 1);
        TTI_SFPMAD(2, 3, 10, 2, 0);
        TTI_SFPMULI(0x3f00, 4, 0);
        TTI_SFPMAD(3, 2, 3, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPMUL(2, 4, 9, 2, 0);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPSETSGN(0, 2, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_sigmoid_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_sigmoid_bf16_scheduled();
}

}  // namespace ckernel::sfpu
