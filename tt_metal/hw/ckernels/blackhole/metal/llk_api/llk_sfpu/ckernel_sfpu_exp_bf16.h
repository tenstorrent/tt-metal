// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_exp() {
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
inline void calculate_exp_bf16_scheduled() {
    TTI_SFPLOADI(3, 8, 0x3fb8);
    TTI_SFPLOADI(3, 0xa, 0xaa3b);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPMAD(0, 3, 5, 0, 0);
    TTI_SFPLOADI(2, 0, 0x437f);
    TTI_SFPSWAP(0, 9, 0, 9);
    TTI_SFPSWAP(0, 2, 0, 1);
    TTI_SFPEXEXP(0, 0, 1, 0);
    TTI_SFPEXMAN(0, 0, 0, 0);
    TTI_SFPSHFT(0, 1, 0, 0);
    TTI_SFPLOADI(4, 8, 0xe5c);
    TTI_SFPLOADI(6, 8, 0x1ad5);
    TTI_SFPEXMAN(0, 0, 1, 1);
    TTI_SFPLOADI(4, 0xa, 0x80c4);
    TTI_SFPLOADI(6, 0xa, 0x4733);
    TTI_SFPCAST(1, 1, 0);
    TTI_SFPLOADI(7, 8, 0x2777);
    TTI_SFPMAD(1, 4, 6, 2, 0);
    TTI_SFPLOADI(7, 0xa, 0x4744);
    TTI_SFPLOADI(5, 8, 0x33b1);
    TTI_SFPMAD(2, 1, 7, 2, 0);
    TTI_SFPLOADI(5, 0xa, 0x66c3);
    TTI_SFPMAD(2, 1, 5, 2, 0);
    TTI_SFPLOADI(5, 8, 0x3f80);
    TTI_SFPLOADI(5, 0xa, 0x3a);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 2);
    load_replay_buf<Exec>(0, 16, [] {
        TTI_SFPMAD(0, 3, 5, 0, 0);
        TTI_SFPLOADI(2, 0, 0x437f);
        TTI_SFPSWAP(0, 9, 0, 9);
        TTI_SFPSWAP(0, 2, 0, 1);
        TTI_SFPEXEXP(0, 0, 1, 0);
        TTI_SFPEXMAN(0, 0, 0, 0);
        TTI_SFPSHFT(0, 1, 0, 0);
        TTI_SFPEXMAN(0, 0, 1, 1);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPMAD(1, 4, 6, 2, 0);
        TTI_SFPLOADI(5, 8, 0x33b1);
        TTI_SFPMAD(2, 1, 7, 2, 0);
        TTI_SFPLOADI(5, 0xa, 0x66c3);
        TTI_SFPMAD(2, 1, 5, 2, 0);
        TTI_SFPLOADI(5, 8, 0x3f80);
        TTI_SFPLOADI(5, 0xa, 0x3a);
    });
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 2);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 4);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 4);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 6);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 6);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 8);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 8);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0xa);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0xa);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0xc);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0xc);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0xe);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0xe);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x10);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x10);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x12);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x12);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x14);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x14);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x16);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x16);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x18);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x18);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x1a);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x1a);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x1c);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x1c);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x1e);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x1e);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x20);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x20);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x22);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x22);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x24);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x24);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x26);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x26);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x28);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x28);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x2a);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x2a);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x2c);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x2c);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x2e);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x2e);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x30);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x30);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x32);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x32);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x34);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x34);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x36);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x36);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x38);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x38);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x3a);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x3a);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x3c);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x3c);
    TTI_SFPLOADI(5, 0, 0x42fe);
    TTI_SFPLOAD(0, 0, 7, 0x3e);
    TTI_REPLAY(0, 16, 0, 0);
    TTI_SFPMAD(2, 1, 5, 1, 0);
    TTI_SFPSETEXP(0, 1, 0, 2);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x3e);
}

// Each call records replay slots 0-15, below replay_buf_offset = 16.
// The slots are the compiled call's, the SFPI compiler's own replay records included.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_exp_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_exp_bf16_scheduled();
}

}  // namespace ckernel::sfpu
