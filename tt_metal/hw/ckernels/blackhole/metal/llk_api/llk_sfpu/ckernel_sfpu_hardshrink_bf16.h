// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_hardshrink() {
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
// calculate_simple_forward's instructions in a stall-filling order, constants held in LRegs.
inline void calculate_hardshrink_bf16_scheduled() {
    TTI_SFPLOADI(7, 8, 0x3f00);
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPLOADI(7, 0xa, 0);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(1, 0, 7, 2);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 2);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 4);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 6);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 4);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 6);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 8);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0xa);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 8);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0xa);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xc);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0xe);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0xc);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0xe);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x10);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x12);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x10);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x12);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x14);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x16);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x14);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x16);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x18);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x1a);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x18);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1c);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x1e);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x20);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x22);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x20);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x22);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x24);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x26);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x24);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x26);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x28);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x2a);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x28);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2c);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x2e);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x30);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x32);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x30);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x32);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x34);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x36);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x34);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x36);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x38);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x3a);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x38);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3c);
    TTI_SFPSETSGN(0, 0, 2, 1);
    TTI_SFPLOAD(1, 0, 7, 0x3e);
    TTI_SFPSETSGN(0, 1, 3, 1);
    TTI_SFPLE(0, 7, 2, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLE(0, 7, 3, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3e);
    TTI_SFPENCC(3, 0, 0, 0xa);
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_hardshrink_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_hardshrink_bf16_scheduled();
}

}  // namespace ckernel::sfpu
