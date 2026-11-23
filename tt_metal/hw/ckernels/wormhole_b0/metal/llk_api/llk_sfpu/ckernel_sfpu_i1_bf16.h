// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct I1Bf16Config {
    struct TtDomainActionRecord {
        float bound;
        uint8_t direction;
        uint8_t inclusive;
        uint8_t action_kind;   // 0=constant, 1=identity, 2=affine, 3=class, 4=signed-inf
        uint8_t return_class;  // 0=NaN, 1=+Inf, 2=-Inf, 3=+0, 4=-0
        float value;
        float scale;
        float bias;
    };
};
}  // namespace ckernel::sfpu

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_i1() {
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
inline void calculate_i1_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 32; ++pass) {
        TTI_SFPLOAD(4, 0, 3, 0);
        TTI_SFPSETSGN(0, 4, 2, 1);
        TTI_SFPMOV(0, 2, 3, 2);
        TTI_SFPSHFT(0xfff, 0, 3, 1);
        TTI_SFPLOADI(0, 2, 0xaa3b);
        TTI_SFPLOADI(5, 2, 0x755c);
        TTI_SFPIADD(0, 12, 3, 6);
        TTI_SFPLOADI(0, 8, 0x3fb8);
        TTI_SFPLOADI(5, 8, 0x42f2);
        TTI_SFPMUL(2, 3, 9, 1, 0);
        TTI_SFPMOV(0, 3, 6, 1);
        TTI_SFPMAD(2, 0, 5, 0, 0);
        TTI_SFPLOADI(5, 2, 0x3626);
        TTI_SFPMUL(6, 1, 9, 1, 0);
        TTI_SFPLOADI(5, 8, 0x4010);
        TTI_SFPEXEXP(0, 0, 6, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPLOADI(7, 2, 0x14c9);
        TTI_SFPADD(10, 1, 5, 5, 0);
        TTI_SFPSHFT(0, 6, 0, 0);
        TTI_SFPLOADI(7, 8, 0x4012);
        TTI_SFPEXMAN(0, 0, 6, 1);
        TTI_SFPMAD(1, 5, 7, 1, 0);
        TTI_SFPCAST(6, 5, 0);
        TTI_SFPMUL(3, 1, 9, 3, 0);
        TTI_SFPMULI(0x3400, 5, 0);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPLOADI(3, 0, 0x7fc0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 2, 0xf404);
        TTI_SFPLOADI(6, 2, 0x2bd7);
        TTI_SFPMUL(2, 2, 9, 7, 0);
        TTI_SFPLOADI(1, 8, 0x35aa);
        TTI_SFPLOADI(6, 8, 0x381e);
        TTI_SFPMAD(1, 7, 6, 1, 0);
        TTI_SFPLOADI(6, 2, 0x759a);
        TTI_SFPLOADI(6, 8, 0x3b35);
        TTI_SFPMAD(1, 7, 6, 1, 0);
        TTI_SFPLOADI(6, 2, 0xf36d);
        TTI_SFPLOADI(6, 8, 0x3d7d);
        TTI_SFPMAD(1, 7, 6, 1, 0);
        TTI_SFPLOADI(6, 2, 0xd27);
        TTI_SFPEXEXP(0, 0, 0, 1);
        TTI_SFPLOADI(6, 8, 0x3f00);
        TTI_SFPMAD(1, 7, 6, 1, 0);
        TTI_SFPLOADI(6, 2, 0x9381);
        TTI_SFPLOADI(7, 2, 0xbb4c);
        TTI_SFPLOADI(6, 8, 0x40e9);
        TTI_SFPLOADI(7, 8, 0x4165);
        TTI_SFPMAD(6, 5, 7, 6, 0);
        TTI_SFPLOADI(7, 2, 0xa390);
        TTI_SFPLOADI(7, 8, 0x41ae);
        TTI_SFPMAD(6, 5, 7, 5, 0);
        TTI_SFPLOADI(6, 2, 0x5b42);
        TTI_SFPMUL(3, 3, 9, 7, 0);
        TTI_SFPLOADI(6, 8, 0xbeca);
        TTI_SFPMUL(5, 3, 9, 3, 0);
        TTI_SFPMUL(7, 6, 9, 5, 0);
        TTI_SFPLOADI(6, 0, 0x40c0);
        TTI_SFPMAD(5, 3, 3, 3, 0);
        TTI_SFPMAD(11, 2, 6, 5, 0);
        TTI_SFPEXEXP(0, 3, 6, 1);
        TTI_SFPIADD(0, 6, 0, 4);
        TTI_SFPIADD(0xf81, 0, 0, 5);
        TTI_SFPMUL(2, 1, 9, 1, 0);
        TTI_SFPSETEXP(0, 3, 0, 0);
        TTI_SFPSETCC(0, 5, 0, 4);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0x42b8);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPMOV(0, 4, 1, 2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPDIVP2(0x7e, 2, 2, 1);
        TTI_SFPLOADI(3, 0, 0x3fff);
        TTI_SFPSETSGN(0, 0, 1, 0);
        TTI_SFPIADD(0, 2, 3, 6);
        TTI_SFPSETCC(0, 3, 0, 6);
        TTI_SFPSETMAN(0, 4, 1, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(0, 0, 0x42b8);
        TTI_SFPMAD(11, 0, 4, 0, 0);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMOV(0, 2, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(2, 0, 0xc2b8);
        TTI_SFPMAD(11, 4, 2, 4, 0);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPSETCC(0, 4, 0, 4);
        TTI_SFPMOV(0, 0, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(0, 6, 3, 0);
        TTI_SFPLOADI(2, 2, 0x80ff);
        TTI_SFPXOR(0, 2, 0, 0);
        TTI_SFPLOADI(2, 2, 0x80ff);
        TTI_SFPAND(0, 2, 0, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPLOADI(1, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
inline void calculate_i1_bf16_body() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_i1_bf16_scheduled();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_i1_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x5f1110a0u);
    calculate_i1_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
}

}  // namespace ckernel::sfpu
