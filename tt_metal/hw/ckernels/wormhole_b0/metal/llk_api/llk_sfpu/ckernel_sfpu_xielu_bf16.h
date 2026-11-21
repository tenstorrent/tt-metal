// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct XieluBf16Config {
    static constexpr uint32_t kDegree[] = {1, 9, 2, 2};
    static constexpr uint32_t kLutBits[] = {
        0xff7fffffu, 0xc0e00000u, 0xb58637bdu, 0x00000001u, 0x5f8f0d18u, 0xbf4ccccdu, 0xbe99999au, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xb0ffd6d4u,
        0x3f000000u, 0x3eccd1ccu, 0x3e089a94u, 0x3d086efdu, 0x3bd662c9u, 0x3a83559fu, 0x38e90000u, 0x37010000u,
        0x34820000u, 0xb556bf8eu, 0xbe99999au, 0x3b0fa562u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3f000000u, 0x3f4ccccdu, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_xielu() {
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
inline void calculate_xielu_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        TTI_SFPLOAD(1, 0, 3, 0);
        TTI_SFPLOADI(2, 0, 0xff80);
        TTI_SFPMAD(11, 2, 1, 2, 0);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPLOADI(1, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPLOADI(4, 0, 0x3482);
        TTI_SFPLOADI(3, 0, 0x3701);
        TTI_SFPMAD(11, 0, 2, 0, 0);
        TTI_SFPMAD(4, 1, 3, 5, 0);
        TTI_SFPLOADI(6, 0, 0x38e9);
        TTI_SFPLOADI(7, 2, 0x559f);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(5, 1, 6, 0, 0);
        TTI_SFPLOADI(7, 8, 0x3a83);
        TTI_SFPLOADI(5, 2, 0x62c9);
        TTI_SFPMAD(0, 1, 7, 0, 0);
        TTI_SFPMAD(4, 2, 3, 4, 0);
        TTI_SFPLOADI(5, 8, 0x3bd6);
        TTI_SFPLOADI(3, 2, 0x6efd);
        TTI_SFPMAD(0, 1, 5, 0, 0);
        TTI_SFPMAD(4, 2, 6, 4, 0);
        TTI_SFPLOADI(3, 8, 0x3d08);
        TTI_SFPLOADI(6, 2, 0x9a94);
        TTI_SFPMAD(0, 1, 3, 0, 0);
        TTI_SFPMAD(4, 2, 7, 4, 0);
        TTI_SFPLOADI(6, 8, 0x3e08);
        TTI_SFPLOADI(7, 2, 0xd1cc);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPMAD(4, 2, 5, 4, 0);
        TTI_SFPLOADI(7, 8, 0x3ecc);
        TTI_SFPMAD(0, 1, 7, 0, 0);
        TTI_SFPMAD(4, 2, 3, 4, 0);
        TTI_SFPLOADI(3, 0, 0x3f00);
        TTI_SFPLOADI(5, 2, 0x999a);
        TTI_SFPMAD(0, 1, 3, 0, 0);
        TTI_SFPMAD(4, 2, 6, 4, 0);
        TTI_SFPLOADI(6, 2, 0xcccd);
        TTI_SFPLOADI(5, 8, 0xbe99);
        TTI_SFPLOADI(6, 8, 0xbf4c);
        TTI_SFPMAD(4, 2, 7, 4, 0);
        TTI_SFPLOADI(7, 2, 0xd6d4);
        TTI_SFPLOADI(7, 8, 0xb0ff);
        TTI_SFPMAD(0, 1, 7, 0, 0);
        TTI_SFPMAD(4, 2, 3, 3, 0);
        TTI_SFPLOADI(4, 0, 0xc0e0);
        TTI_SFPMAD(3, 2, 7, 3, 0);
        TTI_SFPMAD(5, 1, 6, 7, 0);
        TTI_SFPMAD(5, 2, 6, 5, 0);
        TTI_SFPMAD(11, 4, 1, 6, 0);
        TTI_SFPMAD(11, 4, 2, 4, 0);
        TTI_SFPSETCC(0, 6, 0, 4);
        TTI_SFPMOV(0, 0, 7, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(0, 2, 0xa562);
        TTI_SFPLOADI(6, 2, 0x999a);
        TTI_SFPLOADI(0, 8, 0x3b0f);
        TTI_SFPLOADI(6, 8, 0xbe99);
        TTI_SFPSETCC(0, 4, 0, 4);
        TTI_SFPMOV(0, 5, 4, 2);
        TTI_SFPMOV(0, 3, 4, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(5, 2, 0x37bd);
        TTI_SFPMAD(0, 1, 6, 3, 0);
        TTI_SFPLOADI(5, 8, 0xb586);
        TTI_SFPMAD(0, 2, 6, 0, 0);
        TTI_SFPLOADI(6, 2, 0xbf8e);
        TTI_SFPLOADI(6, 8, 0xb556);
        TTI_SFPMAD(3, 1, 6, 3, 0);
        TTI_SFPMAD(0, 2, 6, 0, 0);
        TTI_SFPMAD(11, 5, 1, 6, 0);
        TTI_SFPMAD(11, 5, 2, 5, 0);
        TTI_SFPSETCC(0, 6, 0, 4);
        TTI_SFPMOV(0, 3, 7, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(3, 2, 0xcccd);
        TTI_SFPLOADI(3, 8, 0x3f4c);
        TTI_SFPLOADI(6, 0, 0x3f00);
        TTI_SFPSETCC(0, 5, 0, 4);
        TTI_SFPMOV(0, 0, 4, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(3, 1, 6, 5, 0);
        TTI_SFPMOV(0, 7, 0, 2);
        TTI_SFPMAD(5, 1, 9, 5, 0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPMOV(0, 5, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(3, 2, 6, 1, 0);
        TTI_SFPMOV(0, 4, 3, 2);
        TTI_SFPMAD(1, 2, 9, 4, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPSETCC(0, 2, 0, 2);
        TTI_SFPMOV(0, 4, 3, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(2, 0, 0xff80);
        TTI_SFPMAD(11, 2, 0, 2, 0);
        TTI_SFPLOADI(1, 0, 0xff80);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPLOADI(0, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 1, 3, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 1, 1);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPLOADI(3, 0, 0xff80);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 3, 0, 1);
        TTI_SFPSTORE(1, 0, 3, 0);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_xielu_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_xielu_bf16_scheduled();
}

}  // namespace ckernel::sfpu
