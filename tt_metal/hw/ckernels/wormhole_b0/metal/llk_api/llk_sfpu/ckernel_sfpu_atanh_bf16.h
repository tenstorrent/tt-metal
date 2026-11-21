// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_atanh() {
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
inline void calculate_atanh_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPLOADI(3, 2, 0x10b);
        TTI_SFPLOADI(4, 2, 0xf85f);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPLOADI(3, 8, 0x3f6e);
        TTI_SFPLOADI(4, 8, 0xbff6);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(7, 2, 0x8b65);
        TTI_SFPMAD(3, 1, 10, 3, 0);
        TTI_SFPLOADI(0, 2, 0x8d7c);
        TTI_SFPMOV(0, 3, 5, 2);
        TTI_SFPSETMAN(0, 11, 5, 0);
        TTI_SFPMAD(12, 5, 13, 4, 0);
        TTI_SFPLOADI(7, 8, 0x3e06);
        TTI_SFPLOADI(0, 8, 0xbe3e);
        TTI_SFPMAD(4, 5, 14, 4, 0);
        TTI_SFPLOADI(6, 2, 0xf154);
        TTI_SFPMAD(7, 1, 0, 0, 0);
        TTI_SFPLOADI(6, 8, 0x3e42);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPMAD(5, 4, 10, 6, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(4, 6, 4, 4, 0);
        TTI_SFPLOADI(6, 2, 0xcd4f);
        TTI_SFPLOADI(6, 8, 0x3eeb);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPLOADI(6, 2, 0x2d0b);
        TTI_SFPLOADI(6, 8, 0xbfcc);
        TTI_SFPMAD(5, 4, 10, 5, 0);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPNOT(0, 3, 6, 0);
        TTI_SFPSETMAN(0, 6, 6, 1);
        TTI_SFPMULI(0x3f00, 6, 0);
        TTI_SFPMAD(4, 5, 4, 4, 0);
        TTI_SFPMAD(0, 1, 10, 0, 0);
        TTI_SFPMUL(4, 6, 9, 1, 0);
        TTI_SFPMUL(0, 2, 9, 0, 0);
        TTI_SFPSETSGN(0, 2, 4, 1);
        TTI_SFPSETSGN(0, 1, 3, 0);
        TTI_SFPMAD(11, 4, 10, 1, 0);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMAD(11, 10, 4, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPLOADI(1, 0, 0x7f80);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPSETSGN(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPLOADI(3, 2, 0x10b);
        TTI_SFPLOADI(4, 2, 0xf85f);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPLOADI(3, 8, 0x3f6e);
        TTI_SFPLOADI(4, 8, 0xbff6);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(0, 2, 0x8d7c);
        TTI_SFPMAD(3, 1, 10, 3, 0);
        TTI_SFPLOADI(0, 8, 0xbe3e);
        TTI_SFPMOV(0, 3, 5, 2);
        TTI_SFPSETMAN(0, 11, 5, 0);
        TTI_SFPMAD(12, 5, 13, 4, 0);
        TTI_SFPLOADI(6, 2, 0xf154);
        TTI_SFPMAD(4, 5, 14, 4, 0);
        TTI_SFPMAD(7, 1, 0, 0, 0);
        TTI_SFPLOADI(6, 8, 0x3e42);
        TTI_SFPMAD(5, 4, 10, 7, 0);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPLOADI(6, 2, 0xcd4f);
        TTI_SFPLOADI(6, 8, 0x3eeb);
        TTI_SFPMAD(4, 7, 4, 4, 0);
        TTI_SFPLOADI(7, 2, 0x2d0b);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPNOT(0, 3, 6, 0);
        TTI_SFPLOADI(7, 8, 0xbfcc);
        TTI_SFPMAD(5, 4, 10, 5, 0);
        TTI_SFPSETMAN(0, 6, 6, 1);
        TTI_SFPMAD(0, 1, 7, 0, 0);
        TTI_SFPMULI(0x3f00, 6, 0);
        TTI_SFPMAD(4, 5, 4, 4, 0);
        TTI_SFPMAD(0, 1, 10, 0, 0);
        TTI_SFPMUL(4, 6, 9, 1, 0);
        TTI_SFPMUL(0, 2, 9, 0, 0);
        TTI_SFPSETSGN(0, 2, 4, 1);
        TTI_SFPSETSGN(0, 1, 3, 0);
        TTI_SFPMAD(11, 4, 10, 1, 0);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(0, 0, 0x7fc0);
        TTI_SFPCOMPC(0, 0, 0, 0);
        TTI_SFPMAD(11, 10, 4, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPLOADI(1, 0, 0x7f80);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPSETSGN(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
inline void calculate_atanh_bf16_body() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_atanh_bf16_scheduled();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_atanh_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x3fba2e90u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x4007c1f2u);
    calculate_atanh_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x33b17218u);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x3eba2000u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0xbf004000u);
}

}  // namespace ckernel::sfpu
