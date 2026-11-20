// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_abs() {
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
inline void calculate_abs_bf16_scheduled() {
    TTI_SFPLOAD(0, 0, 3, 0);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0);
    TTI_SFPLOAD(1, 0, 3, 2);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 2);
    TTI_SFPLOAD(0, 0, 3, 4);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 4);
    TTI_SFPLOAD(1, 0, 3, 6);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 6);
    TTI_SFPLOAD(0, 0, 3, 8);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 8);
    TTI_SFPLOAD(1, 0, 3, 0xa);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0xa);
    TTI_SFPLOAD(0, 0, 3, 0xc);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0xc);
    TTI_SFPLOAD(1, 0, 3, 0xe);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0xe);
    TTI_SFPLOAD(0, 0, 3, 0x10);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x10);
    TTI_SFPLOAD(1, 0, 3, 0x12);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x12);
    TTI_SFPLOAD(0, 0, 3, 0x14);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x14);
    TTI_SFPLOAD(1, 0, 3, 0x16);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x16);
    TTI_SFPLOAD(0, 0, 3, 0x18);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x18);
    TTI_SFPLOAD(1, 0, 3, 0x1a);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x1a);
    TTI_SFPLOAD(0, 0, 3, 0x1c);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x1c);
    TTI_SFPLOAD(1, 0, 3, 0x1e);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x1e);
    TTI_SFPLOAD(0, 0, 3, 0x20);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x20);
    TTI_SFPLOAD(1, 0, 3, 0x22);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x22);
    TTI_SFPLOAD(0, 0, 3, 0x24);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x24);
    TTI_SFPLOAD(1, 0, 3, 0x26);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x26);
    TTI_SFPLOAD(0, 0, 3, 0x28);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x28);
    TTI_SFPLOAD(1, 0, 3, 0x2a);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x2a);
    TTI_SFPLOAD(0, 0, 3, 0x2c);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x2c);
    TTI_SFPLOAD(1, 0, 3, 0x2e);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x2e);
    TTI_SFPLOAD(0, 0, 3, 0x30);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x30);
    TTI_SFPLOAD(1, 0, 3, 0x32);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x32);
    TTI_SFPLOAD(0, 0, 3, 0x34);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x34);
    TTI_SFPLOAD(1, 0, 3, 0x36);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x36);
    TTI_SFPLOAD(0, 0, 3, 0x38);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x38);
    TTI_SFPLOAD(1, 0, 3, 0x3a);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x3a);
    TTI_SFPLOAD(0, 0, 3, 0x3c);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPSTORE(0, 0, 3, 0x3c);
    TTI_SFPLOAD(1, 0, 3, 0x3e);
    TTI_SFPABS(0, 1, 1, 1);
    TTI_SFPSTORE(1, 0, 3, 0x3e);
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_abs_bf16() {
    static_assert(ITERATIONS == 32, "the BF16 kernel runs as one whole-tile call");
    calculate_abs_bf16_scheduled();
}

}  // namespace ckernel::sfpu
