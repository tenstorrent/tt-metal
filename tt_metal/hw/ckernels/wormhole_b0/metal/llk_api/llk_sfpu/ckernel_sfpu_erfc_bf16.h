// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct ErfcBf16Config {
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
inline bool bf16_dest_erfc() {
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
inline void calculate_erfc_bf16_scheduled() {
    TTI_SFPLOADI(7, 2, 0x80ff);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 32; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPSETSGN(0, 2, 3, 1);
        TTI_SFPLOADI(0, 0, 0x4114);
        TTI_SFPSWAP(0, 0, 3, 1);
        TTI_SFPMUL(3, 3, 9, 0, 0);
        TTI_SFPLOADI(1, 2, 0xaa3b);
        TTI_SFPLOADI(4, 2, 0x51e0);
        TTI_SFPMOV(0, 0, 0, 1);
        TTI_SFPLOADI(1, 8, 0x3fb8);
        TTI_SFPLOADI(5, 0, 0x42fe);
        TTI_SFPLOADI(4, 8, 0x3f16);
        TTI_SFPMAD(0, 1, 5, 0, 0);
        TTI_SFPLOADI(5, 2, 0xeb0b);
        TTI_SFPMAD(4, 3, 10, 1, 0);
        TTI_SFPLOADI(5, 8, 0x3f1c);
        TTI_SFPEXEXP(0, 0, 4, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPMAD(1, 3, 5, 1, 0);
        TTI_SFPSHFT(0, 4, 0, 0);
        TTI_SFPEXMAN(0, 0, 4, 1);
        TTI_SFPCAST(4, 4, 0);
        TTI_SFPLOADI(5, 2, 0x879);
        TTI_SFPLOADI(6, 2, 0x98ec);
        TTI_SFPMULI(0x3400, 4, 0);
        TTI_SFPLOADI(5, 8, 0x3da0);
        TTI_SFPLOADI(6, 8, 0x3e67);
        TTI_SFPMAD(5, 4, 6, 5, 0);
        TTI_SFPLOADI(6, 2, 0x107d);
        TTI_SFPLOADI(6, 8, 0x3f32);
        TTI_SFPMAD(5, 4, 6, 5, 0);
        TTI_SFPMOV(0, 1, 6, 2);
        TTI_SFPSETMAN(0, 11, 6, 0);
        TTI_SFPMAD(5, 4, 10, 4, 0);
        TTI_SFPMAD(12, 6, 13, 5, 0);
        TTI_SFPEXEXP(0, 0, 0, 1);
        TTI_SFPMAD(5, 6, 14, 5, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(6, 5, 10, 6, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(5, 6, 5, 5, 0);
        TTI_SFPNOT(0, 1, 6, 0);
        TTI_SFPSETMAN(0, 6, 6, 1);
        TTI_SFPMULI(0x3f00, 6, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(5, 6, 9, 5, 0);
        TTI_SFPLOADI(6, 2, 0x9f6d);
        TTI_SFPLOADI(6, 8, 0x3ea7);
        TTI_SFPSETSGN(0, 5, 1, 0);
        TTI_SFPLOADI(5, 2, 0xb49c);
        TTI_SFPLOADI(5, 8, 0x3f1c);
        TTI_SFPMAD(6, 3, 5, 3, 0);
        TTI_SFPEXEXP(0, 4, 5, 1);
        TTI_SFPIADD(0, 5, 0, 4);
        TTI_SFPIADD(0xf81, 0, 0, 5);
        TTI_SFPMUL(3, 1, 9, 1, 0);
        TTI_SFPSETEXP(0, 4, 0, 0);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPLOADI(1, 0, 0x4118);
        TTI_SFPMOV(0, 0, 3, 1);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPADDI(0x4000, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 1, 2, 1, 0);
        TTI_SFPLOADI(3, 0, 0xc0a0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 3, 2, 1, 0);
        TTI_SFPLOADI(3, 0, 0x4000);
        TTI_SFPLOAD(2, 6, 3, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 2, 1, 2);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(3, 2, 0xff);
        TTI_SFPXOR(0, 3, 1, 0);
        TTI_SFPAND(0, 7, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPMOV(0, 2, 1, 2);
        TTI_SFPLOADI(0, 0, 0x2c4e);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPXOR(0, 7, 1, 0);
        TTI_SFPAND(0, 7, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPLOADI(3, 2, 0x7f00);
        TTI_SFPMOV(0, 2, 1, 2);
        TTI_SFPAND(0, 3, 1, 0);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPIADD(0xf01, 2, 1, 5);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPMOV(0, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_erfc_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_erfc_bf16_scheduled();
}

}  // namespace ckernel::sfpu
