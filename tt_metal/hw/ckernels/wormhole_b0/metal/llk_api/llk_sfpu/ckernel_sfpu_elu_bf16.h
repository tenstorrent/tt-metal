// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct EluBf16Config {
    static constexpr uint32_t kDegree[] = {8, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc0c80000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3f800000u, 0x3effdb9eu, 0x3e295070u,
        0x3d22c6dbu, 0x3be47be7u, 0x3a5c41d7u, 0x38800000u, 0x36050000u, 0x00000000u, 0x3f800000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_elu() {
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
inline void calculate_elu_bf16_scheduled() {
    TTI_SFPLOADI(7, 0, 0xc0c8);
    TTI_SFPLOADI(5, 2, 0xc6db);
    TTI_SFPLOADI(5, 8, 0x3d22);
    TTI_SFPLOADI(6, 2, 0x5070);
    TTI_SFPLOADI(6, 8, 0x3e29);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMAD(11, 7, 2, 0, 0);
        TTI_SFPLOADI(1, 0, 0x3605);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 0, 0xc0c8);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(4, 0, 0x3880);
        TTI_SFPLOAD(0, 0, 3, 2);
        TTI_SFPMAD(11, 7, 0, 3, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPLOADI(0, 0, 0xc0c8);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(1, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPLOADI(4, 2, 0x41d7);
        TTI_SFPLOADI(4, 8, 0x3a5c);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPLOADI(4, 2, 0x7be7);
        TTI_SFPLOADI(4, 8, 0x3be4);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPLOADI(4, 2, 0xdb9e);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3eff);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 5, 1, 0);
        TTI_SFPMAD(3, 2, 10, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPADD(10, 2, 9, 4, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMOV(0, 3, 2, 2);
        TTI_SFPMOV(0, 4, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPADD(10, 0, 9, 4, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMOV(0, 4, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 11, 2, 0, 0);
        TTI_SFPMAD(11, 11, 1, 3, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPMOV(0, 11, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPMOV(0, 11, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 0, 1);
        TTI_SFPSTORE(2, 0, 3, 0);
        TTI_SFPSTORE(0, 0, 3, 2);
        TTI_SFPLOAD(2, 0, 3, 4);
        TTI_SFPMAD(11, 7, 2, 0, 0);
        TTI_SFPLOADI(1, 0, 0x3605);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 0, 0xc0c8);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(4, 0, 0x3880);
        TTI_SFPLOAD(0, 0, 3, 6);
        TTI_SFPMAD(11, 7, 0, 3, 0);
        TTI_SFPNOP;
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPLOADI(0, 0, 0xc0c8);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(1, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPLOADI(4, 2, 0x41d7);
        TTI_SFPLOADI(4, 8, 0x3a5c);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPLOADI(4, 2, 0x7be7);
        TTI_SFPLOADI(4, 8, 0x3be4);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPLOADI(4, 2, 0xdb9e);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3eff);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(1, 0, 5, 1, 0);
        TTI_SFPMAD(3, 2, 10, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPMAD(1, 0, 4, 1, 0);
        TTI_SFPADD(10, 2, 9, 4, 0);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPMOV(0, 3, 2, 2);
        TTI_SFPMOV(0, 4, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPADD(10, 0, 9, 4, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPSETCC(0, 0, 0, 4);
        TTI_SFPMOV(0, 4, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMAD(11, 11, 2, 0, 0);
        TTI_SFPMAD(11, 11, 1, 3, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPMOV(0, 11, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPMOV(0, 11, 1, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 0, 1);
        TTI_SFPSTORE(2, 0, 3, 4);
        TTI_SFPSTORE(0, 0, 3, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_elu_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_elu_bf16_scheduled();
}

}  // namespace ckernel::sfpu
