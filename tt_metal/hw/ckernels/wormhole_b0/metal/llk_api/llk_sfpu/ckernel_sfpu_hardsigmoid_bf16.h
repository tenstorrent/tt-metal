// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_hardsigmoid() {
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
inline void calculate_hardsigmoid_bf16_scheduled() {
    TTI_SFPLOADI(6, 8, 0x3e2a);
    TTI_SFPLOADI(7, 8, 0x3f00);
    TTI_SFPLOADI(6, 0xa, 0xaaab);
    TTI_SFPLOADI(7, 0xa, 0);
    TTI_SFPLOAD(0, 0, 3, 0);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPLOAD(1, 0, 3, 2);
    TTI_REPLAY(0, 7, 1, 1);
    TTI_SFPMAD(6, 1, 7, 3, 0);
    TTI_SFPSWAP(0, 9, 2, 9);
    TTI_SFPSWAP(0, 10, 2, 1);
    TTI_SFPSWAP(0, 9, 3, 9);
    TTI_SFPSWAP(0, 10, 3, 1);
    TTI_SFP_STOCH_RND(0, 0, 0, 2, 2, 1);
    TTI_SFP_STOCH_RND(0, 0, 0, 3, 3, 1);
    TTI_SFPSTORE(2, 0, 3, 0);
    TTI_SFPLOAD(0, 0, 3, 4);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 2);
    TTI_SFPLOAD(1, 0, 3, 6);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 4);
    TTI_SFPLOAD(0, 0, 3, 8);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 6);
    TTI_SFPLOAD(1, 0, 3, 0xa);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 8);
    TTI_SFPLOAD(0, 0, 3, 0xc);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0xa);
    TTI_SFPLOAD(1, 0, 3, 0xe);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0xc);
    TTI_SFPLOAD(0, 0, 3, 0x10);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0xe);
    TTI_SFPLOAD(1, 0, 3, 0x12);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x10);
    TTI_SFPLOAD(0, 0, 3, 0x14);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x12);
    TTI_SFPLOAD(1, 0, 3, 0x16);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x14);
    TTI_SFPLOAD(0, 0, 3, 0x18);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x16);
    TTI_SFPLOAD(1, 0, 3, 0x1a);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x18);
    TTI_SFPLOAD(0, 0, 3, 0x1c);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x1a);
    TTI_SFPLOAD(1, 0, 3, 0x1e);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x1c);
    TTI_SFPLOAD(0, 0, 3, 0x20);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x1e);
    TTI_SFPLOAD(1, 0, 3, 0x22);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x20);
    TTI_SFPLOAD(0, 0, 3, 0x24);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x22);
    TTI_SFPLOAD(1, 0, 3, 0x26);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x24);
    TTI_SFPLOAD(0, 0, 3, 0x28);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x26);
    TTI_SFPLOAD(1, 0, 3, 0x2a);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x28);
    TTI_SFPLOAD(0, 0, 3, 0x2c);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x2a);
    TTI_SFPLOAD(1, 0, 3, 0x2e);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x2c);
    TTI_SFPLOAD(0, 0, 3, 0x30);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x2e);
    TTI_SFPLOAD(1, 0, 3, 0x32);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x30);
    TTI_SFPLOAD(0, 0, 3, 0x34);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x32);
    TTI_SFPLOAD(1, 0, 3, 0x36);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x34);
    TTI_SFPLOAD(0, 0, 3, 0x38);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x36);
    TTI_SFPLOAD(1, 0, 3, 0x3a);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x38);
    TTI_SFPLOAD(0, 0, 3, 0x3c);
    TTI_SFPMAD(6, 0, 7, 2, 0);
    TTI_SFPSTORE(3, 0, 3, 0x3a);
    TTI_SFPLOAD(1, 0, 3, 0x3e);
    TTI_REPLAY(0, 7, 0, 0);
    TTI_SFPSTORE(2, 0, 3, 0x3c);
    TTI_SFPSTORE(3, 0, 3, 0x3e);
}

// Each call records replay slots 0-6, below replay_buf_offset = 16.
// The slots are the compiled call's, the SFPI compiler's own replay records included.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_hardsigmoid_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_hardsigmoid_bf16_scheduled();
}

}  // namespace ckernel::sfpu
