// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SeluBf16Config {
    static constexpr uint32_t kDegree[] = {8, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc0c20000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3fe10966u, 0x3f6118f6u, 0x3e95c291u,
        0x3d927a57u, 0x3c54e4eau, 0x3ad80000u, 0x3905727fu, 0x36940000u, 0x00000000u, 0x3f867d5fu,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_selu() {
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
inline void calculate_selu_bf16_scheduled() {
    TTI_SFPLOADI(7, 0, 0xc0c2);
    TTI_SFPLOADI(6, 2, 0xc291);
    TTI_SFPLOADI(6, 8, 0x3e95);
    TTI_SFPLOADI(5, 2, 0x7d5f);
    TTI_SFPLOADI(5, 8, 0x3f86);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMAD(11, 7, 2, 0, 0);
        TTI_SFPLOADI(4, 2, 0x727f);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 0, 0xc0c2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 0, 3, 2);
        TTI_SFPLOADI(0, 0, 0x3694);
        TTI_SFPLOADI(4, 8, 0x3905);
        TTI_SFPMAD(11, 7, 1, 3, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPLOADI(1, 0, 0xc0c2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(0, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 0, 0x3ad8);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0xe4ea);
        TTI_SFPLOADI(4, 8, 0x3c54);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x7a57);
        TTI_SFPLOADI(4, 8, 0x3d92);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x18f6);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3f61);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x966);
        TTI_SFPLOADI(4, 8, 0x3fe1);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPMAD(5, 2, 9, 4, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMOV(0, 4, 3, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(0, 1, 9, 0, 0);
        TTI_SFPMAD(5, 1, 9, 4, 0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPMOV(0, 4, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xbfe1);
        TTI_SFPMAD(11, 1, 3, 1, 0);
        TTI_SFPLOADI(2, 0, 0xbfe1);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(3, 0, 0xbfe1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 2, 0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 3, 3, 1);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(0, 0, 0xbfe1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(3, 0, 3, 0);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_SFPLOAD(2, 0, 3, 4);
        TTI_SFPMAD(11, 7, 2, 0, 0);
        TTI_SFPLOADI(4, 2, 0x727f);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 0, 0xc0c2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 0, 3, 6);
        TTI_SFPLOADI(0, 0, 0x3694);
        TTI_SFPLOADI(4, 8, 0x3905);
        TTI_SFPMAD(11, 7, 1, 3, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPLOADI(1, 0, 0xc0c2);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(0, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 0, 0x3ad8);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0xe4ea);
        TTI_SFPLOADI(4, 8, 0x3c54);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x7a57);
        TTI_SFPLOADI(4, 8, 0x3d92);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x18f6);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3f61);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 6, 0, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x966);
        TTI_SFPLOADI(4, 8, 0x3fe1);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(0, 1, 4, 0, 0);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPMAD(5, 2, 9, 4, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMOV(0, 4, 3, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(0, 1, 9, 0, 0);
        TTI_SFPMAD(5, 1, 9, 4, 0);
        TTI_SFPSETCC(0, 1, 0, 4);
        TTI_SFPMOV(0, 4, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xbfe1);
        TTI_SFPMAD(11, 1, 3, 1, 0);
        TTI_SFPLOADI(2, 0, 0xbfe1);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(3, 0, 0xbfe1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 2, 0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 3, 3, 1);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPLOADI(0, 0, 0xbfe1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(3, 0, 3, 4);
        TTI_SFPSTORE(0, 0, 3, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_selu_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_selu_bf16_scheduled();
}

}  // namespace ckernel::sfpu
