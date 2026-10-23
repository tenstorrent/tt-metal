// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct TanhshrinkBf16Config {
    static constexpr uint32_t kDegree = 8u;
    // Fit: minimax degree-8 polynomial on [0, 3], 1 segment, max pure (continuous) ULP 0.781.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u,
        0x00000000u,
        0x00000000u,
        0x3eaa738eu,
        0x3c52ec0cu,
        0xbe4a3e90u,
        0x3dec2feeu,
        0xbce1764au,
        0x3b20a66au};
    static constexpr uint32_t kBoundBits = 0x40400000u;
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBiasBits = 0xbf800000u;
    static constexpr uint32_t kLeftBiasBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 32u;
    static constexpr uint32_t kRawEqualValue = 0x80ffu;
    static constexpr uint32_t kRepairedPatterns = 127u;
    static constexpr uint32_t kMacroSequenceBits = 0x13850000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_tanhshrink_bf16_cascade_signed_abs_affine.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_tanhshrink() {
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
// ckernel::sfpu::bf16::tanhshrink::calculate_signed_abs_affine<ckernel::sfpu::TanhshrinkBf16Config, 32> in a
// stall-filling order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every
// BF16 input; other ITERATIONS run the compiled body itself.
inline void calculate_tanhshrink_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        TTI_SFPLOAD(0, 0, 7, 0);
        TTI_SFPLOADI(3, 8, 0x4040);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPLOADI(3, 0xa, 0);
        TTI_SFPLOADI(2, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPSWAP(0, 3, 1, 1);
        TTI_SFPLOADI(2, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(2, 1, 4, 2, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPLOADI(7, 8, 0xbe4a);
        TTI_SFPMAD(2, 1, 4, 2, 0);
        TTI_SFPLOADI(7, 0xa, 0x3e90);
        TTI_SFPLOADI(6, 8, 0x3c52);
        TTI_SFPMAD(2, 1, 7, 2, 0);
        TTI_SFPLOADI(6, 0xa, 0xec0c);
        TTI_SFPLOADI(5, 8, 0x3eaa);
        TTI_SFPMAD(2, 1, 6, 2, 0);
        TTI_SFPLOADI(5, 0xa, 0x738e);
        TTI_SFPMAD(2, 1, 5, 2, 0);
        TTI_SFPSETSGN(0, 0, 3, 1);
        TTI_SFPLOAD(4, 0, 7, 2);
        TTI_SFPMAD(2, 1, 9, 2, 0);
        TTI_SFPMAD(3, 10, 11, 3, 0);
        TTI_SFPMAD(2, 1, 9, 2, 0);
        TTI_SFPMAD(2, 1, 9, 2, 0);
        TTI_SFPLOADI(1, 8, 0x4040);
        TTI_SFPLOADI(1, 0xa, 0);
        TTI_SFPSWAP(0, 3, 2, 9);
        TTI_SFPSETSGN(0, 4, 3, 1);
        TTI_SFPSWAP(0, 1, 3, 1);
        TTI_SFPLOAD(1, 6, 7, 0);
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPLOADI(2, 2, 0x80ff);
        TTI_SFPXOR(0, 2, 1, 0);
        TTI_SFPAND(2, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPSETSGN(0, 0, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(2, 8, 0x3b20);
        TTI_SFPLOADI(1, 8, 0xbce1);
        TTI_SFPLOADI(2, 0xa, 0xa66a);
        TTI_SFPLOADI(1, 0xa, 0x764a);
        TTI_SFPMAD(2, 3, 1, 2, 0);
        TTI_SFPLOADI(1, 8, 0x3dec);
        TTI_SFPLOADI(1, 0xa, 0x2fee);
        TTI_SFPMAD(2, 3, 1, 2, 0);
        TTI_SFPSETSGN(0, 4, 1, 1);
        TTI_SFPMAD(2, 3, 7, 2, 0);
        TTI_SFPMAD(1, 10, 11, 1, 0);
        TTI_SFPMAD(2, 3, 6, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPMAD(2, 3, 5, 2, 0);
        TTI_SFPSTORE(0, 0, 7, 0);
        TTI_SFPMAD(2, 3, 9, 2, 0);
        TTI_SFPLOAD(0, 6, 7, 2);
        TTI_SFPMAD(2, 3, 9, 2, 0);
        TTI_SFPMAD(2, 3, 9, 2, 0);
        TTI_SFPLOAD(3, 0, 7, 4);
        TTI_SFPSWAP(0, 1, 2, 9);
        TTI_SFPLOADI(1, 8, 0x4040);
        TTI_SFPLOADI(1, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 4, 0);
        TTI_SFPSETSGN(0, 3, 2, 1);
        TTI_SFPSWAP(0, 1, 2, 1);
        TTI_SFPLOADI(1, 2, 0x80ff);
        TTI_SFPXOR(0, 1, 0, 0);
        TTI_SFPAND(1, 0, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETCC(0, 0, 0, 2);
        TTI_SFPSETSGN(0, 4, 4, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(0, 8, 0x3b20);
        TTI_SFPLOADI(1, 8, 0xbce1);
        TTI_SFPLOADI(0, 0xa, 0xa66a);
        TTI_SFPLOADI(1, 0xa, 0x764a);
        TTI_SFPMAD(0, 2, 1, 0, 0);
        TTI_SFPLOADI(1, 8, 0x3dec);
        TTI_SFPLOADI(1, 0xa, 0x2fee);
        TTI_SFPMAD(0, 2, 1, 0, 0);
        TTI_SFPSETSGN(0, 3, 1, 1);
        TTI_SFPMAD(0, 2, 7, 0, 0);
        TTI_SFPMAD(1, 10, 11, 1, 0);
        TTI_SFPMAD(0, 2, 6, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 4, 4, 1);
        TTI_SFPMAD(0, 2, 5, 0, 0);
        TTI_SFPSTORE(4, 0, 7, 2);
        TTI_SFPMAD(0, 2, 9, 0, 0);
        TTI_SFPLOAD(4, 6, 7, 4);
        TTI_SFPMAD(0, 2, 9, 0, 0);
        TTI_SFPMAD(0, 2, 9, 2, 0);
        TTI_SFPLOAD(0, 0, 7, 6);
        TTI_SFPSWAP(0, 1, 2, 9);
        TTI_SFPLOADI(1, 8, 0x4040);
        TTI_SFPLOADI(1, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 3, 0);
        TTI_SFPSETSGN(0, 0, 2, 1);
        TTI_SFPSWAP(0, 1, 2, 1);
        TTI_SFPLOADI(1, 2, 0x80ff);
        TTI_SFPXOR(0, 1, 4, 0);
        TTI_SFPAND(1, 4, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETCC(0, 4, 0, 2);
        TTI_SFPSETSGN(0, 3, 3, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(1, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPLOADI(1, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(1, 2, 4, 1, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPMAD(1, 2, 4, 1, 0);
        TTI_SFPSETSGN(0, 0, 4, 1);
        TTI_SFPMAD(1, 2, 7, 1, 0);
        TTI_SFPMAD(4, 10, 11, 4, 0);
        TTI_SFPMAD(1, 2, 6, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 3, 3, 1);
        TTI_SFPMAD(1, 2, 5, 1, 0);
        TTI_SFPSTORE(3, 0, 7, 4);
        TTI_SFPMAD(1, 2, 9, 1, 0);
        TTI_SFPLOAD(3, 6, 7, 6);
        TTI_SFPMAD(1, 2, 9, 1, 0);
        TTI_SFPMAD(1, 2, 9, 2, 0);
        TTI_SFPLOAD(1, 0, 7, 8);
        TTI_SFPSWAP(0, 4, 2, 9);
        TTI_SFPLOADI(4, 8, 0x4040);
        TTI_SFPLOADI(4, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPLOADI(4, 2, 0x80ff);
        TTI_SFPXOR(0, 4, 3, 0);
        TTI_SFPAND(4, 3, 4, 1);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETSGN(0, 0, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(3, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPLOADI(3, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPSETSGN(0, 1, 4, 1);
        TTI_SFPMAD(3, 2, 7, 3, 0);
        TTI_SFPMAD(4, 10, 11, 4, 0);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPSTORE(0, 0, 7, 6);
        TTI_SFPMAD(3, 2, 9, 0, 0);
        TTI_SFPLOAD(3, 6, 7, 8);
        TTI_SFPMAD(0, 2, 9, 0, 0);
        TTI_SFPMAD(0, 2, 9, 2, 0);
        TTI_SFPLOAD(0, 0, 7, 0xa);
        TTI_SFPSWAP(0, 4, 2, 9);
        TTI_SFPLOADI(4, 8, 0x4040);
        TTI_SFPLOADI(4, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 1, 0);
        TTI_SFPSETSGN(0, 0, 2, 1);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPLOADI(4, 2, 0x80ff);
        TTI_SFPXOR(0, 4, 3, 0);
        TTI_SFPAND(4, 3, 4, 1);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETSGN(0, 1, 1, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(3, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPLOADI(3, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPSETSGN(0, 0, 4, 1);
        TTI_SFPMAD(3, 2, 7, 3, 0);
        TTI_SFPMAD(4, 10, 11, 4, 0);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPSTORE(1, 0, 7, 8);
        TTI_SFPMAD(3, 2, 9, 1, 0);
        TTI_SFPLOAD(3, 6, 7, 0xa);
        TTI_SFPMAD(1, 2, 9, 1, 0);
        TTI_SFPMAD(1, 2, 9, 2, 0);
        TTI_SFPLOAD(1, 0, 7, 0xc);
        TTI_SFPSWAP(0, 4, 2, 9);
        TTI_SFPLOADI(4, 8, 0x4040);
        TTI_SFPLOADI(4, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPLOADI(4, 2, 0x80ff);
        TTI_SFPXOR(0, 4, 3, 0);
        TTI_SFPAND(4, 3, 4, 1);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETSGN(0, 0, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(3, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPLOADI(3, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPSETSGN(0, 1, 4, 1);
        TTI_SFPMAD(3, 2, 7, 3, 0);
        TTI_SFPMAD(4, 10, 11, 4, 0);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPSTORE(0, 0, 7, 0xa);
        TTI_SFPMAD(3, 2, 9, 0, 0);
        TTI_SFPLOAD(3, 6, 7, 0xc);
        TTI_SFPMAD(0, 2, 9, 0, 0);
        TTI_SFPMAD(0, 2, 9, 2, 0);
        TTI_SFPLOAD(0, 0, 7, 0xe);
        TTI_SFPSWAP(0, 4, 2, 9);
        TTI_SFPLOADI(4, 8, 0x4040);
        TTI_SFPLOADI(4, 0xa, 0);
        TTI_SFPSETSGN(0, 2, 1, 0);
        TTI_SFPSETSGN(0, 0, 2, 1);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPLOADI(4, 2, 0x80ff);
        TTI_SFPXOR(0, 4, 3, 0);
        TTI_SFPAND(4, 3, 4, 1);
        TTI_SFPSETCC(0, 4, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETSGN(0, 1, 1, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPLOADI(3, 8, 0x3b20);
        TTI_SFPLOADI(4, 8, 0xbce1);
        TTI_SFPLOADI(3, 0xa, 0xa66a);
        TTI_SFPLOADI(4, 0xa, 0x764a);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3dec);
        TTI_SFPLOADI(4, 0xa, 0x2fee);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPSETSGN(0, 0, 4, 1);
        TTI_SFPMAD(3, 2, 7, 3, 0);
        TTI_SFPMAD(4, 10, 11, 4, 0);
        TTI_SFPMAD(3, 2, 6, 3, 0);
        TTI_SFPLOAD(6, 6, 7, 0xe);
        TTI_SFPMAD(3, 2, 5, 3, 0);
        TTI_SFPLOADI(5, 2, 0x80ff);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPXOR(0, 5, 6, 0);
        TTI_SFPMAD(3, 2, 9, 3, 0);
        TTI_SFPAND(5, 6, 5, 1);
        TTI_SFPMAD(3, 2, 9, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 1);
        TTI_SFPSWAP(0, 4, 2, 9);
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPSETCC(0, 5, 0, 6);
        TTI_SFPSETCC(0, 6, 0, 2);
        TTI_SFPSETSGN(0, 0, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 2, 1);
        TTI_SFPSTORE(1, 0, 7, 0xc);
        TTI_SFPSTORE(2, 0, 7, 0xe);
        TTI_SFPLOAD(4, 6, 7, 0x10);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_tanhshrink_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_tanhshrink_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::tanhshrink::calculate_signed_abs_affine<ckernel::sfpu::TanhshrinkBf16Config, ITERATIONS>();
    }
}
inline void init_tanhshrink_bf16() {
    ckernel::sfpu::bf16::tanhshrink::init_signed_abs_affine<ckernel::sfpu::TanhshrinkBf16Config>();
}

}  // namespace ckernel::sfpu
