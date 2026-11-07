// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_leaky_relu() {
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
// calculate_slope_max's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_leaky_relu_bf16_scheduled() {
    TTI_SFPLOADI(4, 8, 0x3c23);
    TTI_SFPLOADI(4, 0xa, 0xd70a);
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0);
    TTI_SFPLOAD(1, 0, 7, 2);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 2);
    TTI_SFPLOAD(0, 0, 7, 4);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 4);
    TTI_SFPLOAD(1, 0, 7, 6);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 6);
    TTI_SFPLOAD(0, 0, 7, 8);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 8);
    TTI_SFPLOAD(1, 0, 7, 0xa);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xc);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0xc);
    TTI_SFPLOAD(1, 0, 7, 0xe);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0xe);
    TTI_SFPLOAD(0, 0, 7, 0x10);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x10);
    TTI_SFPLOAD(1, 0, 7, 0x12);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x12);
    TTI_SFPLOAD(0, 0, 7, 0x14);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x14);
    TTI_SFPLOAD(1, 0, 7, 0x16);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x16);
    TTI_SFPLOAD(0, 0, 7, 0x18);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x18);
    TTI_SFPLOAD(1, 0, 7, 0x1a);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1a);
    TTI_SFPLOAD(0, 0, 7, 0x1c);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x1c);
    TTI_SFPLOAD(1, 0, 7, 0x1e);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1e);
    TTI_SFPLOAD(0, 0, 7, 0x20);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x20);
    TTI_SFPLOAD(1, 0, 7, 0x22);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x22);
    TTI_SFPLOAD(0, 0, 7, 0x24);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x24);
    TTI_SFPLOAD(1, 0, 7, 0x26);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x26);
    TTI_SFPLOAD(0, 0, 7, 0x28);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x28);
    TTI_SFPLOAD(1, 0, 7, 0x2a);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2a);
    TTI_SFPLOAD(0, 0, 7, 0x2c);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x2c);
    TTI_SFPLOAD(1, 0, 7, 0x2e);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2e);
    TTI_SFPLOAD(0, 0, 7, 0x30);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x30);
    TTI_SFPLOAD(1, 0, 7, 0x32);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x32);
    TTI_SFPLOAD(0, 0, 7, 0x34);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x34);
    TTI_SFPLOAD(1, 0, 7, 0x36);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x36);
    TTI_SFPLOAD(0, 0, 7, 0x38);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x38);
    TTI_SFPLOAD(1, 0, 7, 0x3a);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3a);
    TTI_SFPLOAD(0, 0, 7, 0x3c);
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMUL(4, 0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 7, 0x3c);
    TTI_SFPLOAD(1, 0, 7, 0x3e);
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3e);
}

// Each call records replay slots 0-5, below replay_buf_offset = 16.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_leaky_relu_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_leaky_relu_bf16_scheduled();
}

}  // namespace ckernel::sfpu
