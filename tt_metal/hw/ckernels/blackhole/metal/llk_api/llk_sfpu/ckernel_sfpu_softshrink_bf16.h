// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftshrinkBf16Config {
    static constexpr uint32_t kThresholdBits = 0x3f000000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_softshrink_bf16_simple_forward.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_softshrink() {
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
// The Blackhole whole-tile call as tt-polynomial-fitter's scheduler issues it (ttpoly/sfpu_schedule.py
// in tenstorrent/tt-polynomial-fitter, not in this repository): the compiled instructions of
// ckernel::sfpu::bf16::softshrink::calculate_simple_forward<ckernel::sfpu::SoftshrinkBf16Config, 32> in a stall-filling
// order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every BF16 input;
// other ITERATIONS run the compiled body itself.
inline void calculate_softshrink_bf16_scheduled() {
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPLOADI(7, 0, 0x3f00);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 2);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 2);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 2);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 4);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 4);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 4);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 6);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 6);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 6);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 8);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 8);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 8);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xa);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xa);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xa);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xc);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xc);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xc);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xe);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xe);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xe);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x10);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x10);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x10);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x12);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x12);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x12);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x14);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x14);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x14);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x16);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x16);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x16);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x18);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x18);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x18);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x20);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x20);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x20);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x22);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x22);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x22);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x24);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x24);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x24);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x26);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x26);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x26);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x28);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x28);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x28);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x30);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x30);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x30);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x32);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x32);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x32);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x34);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x34);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x34);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x36);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x36);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x36);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x38);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x38);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x38);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3e);
    TTI_SFPENCC(3, 0, 0, 0xa);
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_softshrink_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_softshrink_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::softshrink::calculate_simple_forward<ckernel::sfpu::SoftshrinkBf16Config, ITERATIONS>();
    }
}
inline void init_softshrink_bf16() {
    ckernel::sfpu::bf16::softshrink::init_simple_forward<ckernel::sfpu::SoftshrinkBf16Config>();
}

}  // namespace ckernel::sfpu
