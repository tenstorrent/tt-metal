// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_tanh() {
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
inline void calculate_tanh_bf16_scheduled() {
    TTI_SFPLOADI(6, 1, 0xb794);
    TTI_SFPLOADI(7, 1, 0x25ff);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        TTI_SFPLOAD(4, 0, 3, 0);
        TTI_SFPABS(0, 4, 0, 1);
        TTI_SFPLOAD(2, 0, 3, 2);
        TTI_SFPMAD(12, 0, 13, 1, 0);
        TTI_SFPABS(0, 2, 5, 1);
        TTI_SFPMAD(12, 5, 13, 3, 0);
        TTI_SFPMAD(1, 0, 14, 1, 0);
        TTI_SFPMAD(3, 5, 14, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 5, 6, 3, 0);
        TTI_SFPMAD(1, 0, 7, 1, 0);
        TTI_SFPMAD(3, 5, 7, 3, 0);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPMAD(3, 5, 10, 3, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPMAD(3, 5, 9, 0, 0);
        TTI_SFPSWAP(0, 10, 1, 1);
        TTI_SFPSWAP(0, 10, 0, 1);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 4, 4, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(4, 0, 3, 0);
        TTI_SFPLOAD(4, 0, 3, 4);
        TTI_SFPABS(0, 4, 0, 1);
        TTI_SFPMAD(12, 0, 13, 1, 0);
        TTI_SFPSTORE(2, 0, 3, 2);
        TTI_SFPLOAD(2, 0, 3, 6);
        TTI_SFPABS(0, 2, 5, 1);
        TTI_SFPMAD(12, 5, 13, 3, 0);
        TTI_SFPMAD(1, 0, 14, 1, 0);
        TTI_SFPMAD(3, 5, 14, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 5, 6, 3, 0);
        TTI_SFPMAD(1, 0, 7, 1, 0);
        TTI_SFPMAD(3, 5, 7, 3, 0);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPMAD(3, 5, 10, 3, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPMAD(3, 5, 9, 0, 0);
        TTI_SFPSWAP(0, 10, 1, 1);
        TTI_SFPSWAP(0, 10, 0, 1);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 4, 4, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(4, 0, 3, 4);
        TTI_SFPLOAD(4, 0, 3, 8);
        TTI_SFPABS(0, 4, 0, 1);
        TTI_SFPMAD(12, 0, 13, 1, 0);
        TTI_SFPSTORE(2, 0, 3, 6);
        TTI_SFPLOAD(2, 0, 3, 0xa);
        TTI_SFPABS(0, 2, 5, 1);
        TTI_SFPMAD(12, 5, 13, 3, 0);
        TTI_SFPMAD(1, 0, 14, 1, 0);
        TTI_SFPMAD(3, 5, 14, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 5, 6, 3, 0);
        TTI_SFPMAD(1, 0, 7, 1, 0);
        TTI_SFPMAD(3, 5, 7, 3, 0);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPMAD(3, 5, 10, 3, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPMAD(3, 5, 9, 0, 0);
        TTI_SFPSWAP(0, 10, 1, 1);
        TTI_SFPSWAP(0, 10, 0, 1);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 4, 4, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(4, 0, 3, 8);
        TTI_SFPLOAD(4, 0, 3, 0xc);
        TTI_SFPABS(0, 4, 0, 1);
        TTI_SFPMAD(12, 0, 13, 1, 0);
        TTI_SFPSTORE(2, 0, 3, 0xa);
        TTI_SFPLOAD(2, 0, 3, 0xe);
        TTI_SFPABS(0, 2, 5, 1);
        TTI_SFPMAD(12, 5, 13, 3, 0);
        TTI_SFPMAD(1, 0, 14, 1, 0);
        TTI_SFPMAD(3, 5, 14, 3, 0);
        TTI_SFPMAD(1, 0, 6, 1, 0);
        TTI_SFPMAD(3, 5, 6, 3, 0);
        TTI_SFPMAD(1, 0, 7, 1, 0);
        TTI_SFPMAD(3, 5, 7, 3, 0);
        TTI_SFPMAD(1, 0, 10, 1, 0);
        TTI_SFPMAD(3, 5, 10, 3, 0);
        TTI_SFPMAD(1, 0, 9, 1, 0);
        TTI_SFPMAD(3, 5, 9, 0, 0);
        TTI_SFPSWAP(0, 10, 1, 1);
        TTI_SFPSWAP(0, 10, 0, 1);
        TTI_SFPSETSGN(0, 1, 4, 0);
        TTI_SFPSETSGN(0, 0, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 4, 4, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
        TTI_SFPSTORE(4, 0, 3, 0xc);
        TTI_SFPSTORE(2, 0, 3, 0xe);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
inline void calculate_tanh_bf16_body() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_tanh_bf16_scheduled();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_tanh_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3bb0e52du);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbd8035c9u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e8a1bd5u);
    calculate_tanh_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3bc0919eu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbd887f48u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e905782u);
}

}  // namespace ckernel::sfpu
