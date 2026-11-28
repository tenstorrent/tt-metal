// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel_sfpu_rsqrt_bf16_newton_root.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_rsqrt() {
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
inline void calculate_rsqrt_bf16_scheduled() {
    TTI_SFPLOADI(7, 0, 0x7f80);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOAD(1, 0, 3, 0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPSHFT(0xfff, 0, 0, 1);
        TTI_SFPIADD(0, 12, 0, 6);
        TTI_SFPMUL(1, 0, 9, 2, 0);
        TTI_SFPMOV(0, 0, 5, 1);
        TTI_SFPMUL(5, 2, 9, 2, 0);
        TTI_SFPLOADI(3, 0, 0x3fc0);
        TTI_SFPADD(10, 14, 2, 5, 0);
        TTI_SFPEXEXP(0, 1, 6, 1);
        TTI_SFPMAD(2, 5, 13, 2, 0);
        TTI_SFPLOAD(4, 0, 3, 2);
        TTI_SFPMUL(0, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 6, 0, 6);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMOV(0, 4, 2, 0);
        TTI_SFPSHFT(0xfff, 2, 2, 1);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 1, 5, 2);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0, 12, 2, 6);
        TTI_SFPMUL(4, 2, 9, 6, 0);
        TTI_SFPIADD(0, 7, 5, 6);
        TTI_SFPABS(0, 1, 1, 1);
        TTI_SFPSETCC(0, 5, 0, 2);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMOV(0, 5, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMOV(0, 2, 5, 1);
        TTI_SFPMUL(5, 6, 9, 1, 0);
        TTI_SFPEXEXP(0, 4, 6, 1);
        TTI_SFPADD(10, 14, 1, 5, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPMAD(1, 5, 13, 1, 0);
        TTI_SFPMOV(0, 4, 5, 2);
        TTI_SFPMUL(2, 1, 9, 1, 0);
        TTI_SFPSETCC(0, 6, 0, 6);
        TTI_SFPMUL(1, 3, 9, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 4, 0, 0);
        TTI_SFPLOADI(1, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0, 7, 5, 6);
        TTI_SFPABS(0, 4, 2, 1);
        TTI_SFPSETCC(0, 5, 0, 2);
        TTI_SFPSETCC(0, 2, 0, 2);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMOV(0, 5, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPLOAD(0, 0, 3, 4);
        TTI_SFPMOV(0, 0, 2, 0);
        TTI_SFPSHFT(0xfff, 2, 2, 1);
        TTI_SFPIADD(0, 12, 2, 6);
        TTI_SFPMUL(0, 2, 9, 4, 0);
        TTI_SFPMOV(0, 2, 5, 1);
        TTI_SFPMUL(5, 4, 9, 4, 0);
        TTI_SFPEXEXP(0, 0, 6, 1);
        TTI_SFPADD(10, 14, 4, 5, 0);
        TTI_SFPSTORE(1, 0, 3, 2);
        TTI_SFPMAD(4, 5, 13, 1, 0);
        TTI_SFPLOAD(4, 0, 3, 6);
        TTI_SFPMUL(2, 1, 9, 1, 0);
        TTI_SFPSETCC(0, 6, 0, 6);
        TTI_SFPMUL(1, 3, 9, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMOV(0, 4, 2, 0);
        TTI_SFPSHFT(0xfff, 2, 2, 1);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPMOV(0, 0, 5, 2);
        TTI_SFPLOADI(1, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0, 12, 2, 6);
        TTI_SFPMUL(4, 2, 9, 6, 0);
        TTI_SFPIADD(0, 7, 5, 6);
        TTI_SFPABS(0, 0, 0, 1);
        TTI_SFPSETCC(0, 5, 0, 2);
        TTI_SFPSETCC(0, 0, 0, 2);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMOV(0, 5, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMOV(0, 2, 5, 1);
        TTI_SFPMUL(5, 6, 9, 0, 0);
        TTI_SFPEXEXP(0, 4, 6, 1);
        TTI_SFPADD(10, 14, 0, 5, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPMAD(0, 5, 13, 0, 0);
        TTI_SFPMOV(0, 4, 5, 2);
        TTI_SFPMUL(2, 0, 9, 0, 0);
        TTI_SFPSETCC(0, 6, 0, 6);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 4, 0, 0);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0, 7, 5, 6);
        TTI_SFPABS(0, 4, 2, 1);
        TTI_SFPSETCC(0, 5, 0, 2);
        TTI_SFPSETCC(0, 2, 0, 2);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMOV(0, 5, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(1, 0, 3, 4);
        TTI_SFPSTORE(0, 0, 3, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_rsqrt_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_rsqrt_bf16_scheduled();
}

}  // namespace ckernel::sfpu
