// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct MishBf16Config {
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0x3fb8aa3bu);
    static constexpr float kBias = __builtin_bit_cast(float, 0x43060000u);
    static constexpr float kNegativeSlope = __builtin_bit_cast(float, 0x3f804189u);
    static constexpr float kPositiveScale = __builtin_bit_cast(float, 0x46ffbe77u);
    static constexpr uint32_t kRawNegativeNanWord = 32640u;
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kCoordinateUpperBf16 = 17172u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_mish_bf16_exponent_alu_product.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_mish() {
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
// ckernel::sfpu::bf16::mish::calculate_exponent_alu_product<ckernel::sfpu::MishBf16Config, 32> in a stall-filling
// order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every BF16 input;
// other ITERATIONS run the compiled body itself.
inline void calculate_mish_bf16_scheduled() {
    TTI_SFPLOADI(6, 8, 0x3f80);
    TTI_SFPLOADI(6, 0xa, 0x3884);
    TTI_SFPLOADI(7, 8, 0xc6ff);
    TTI_SFPLOADI(7, 0xa, 0xbe77);
    TTI_SFPLOADI(5, 2, 0x80ff);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 0);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPARECIP(0, 2, 3, 0);
        TTI_SFPMAD(2, 3, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 3, 9, 3, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(1, 4, 9, 2, 0);
        TTI_SFPMUL(0, 3, 9, 0, 0);
        TTI_SFPMAD(7, 3, 10, 3, 0);
        TTI_SFPMUL(2, 0, 9, 0, 0);
        TTI_SFPMUL(1, 3, 9, 2, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 0, 2, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(0, 6, 7, 0);
        TTI_SFPXOR(0, 5, 0, 0);
        TTI_SFPAND(5, 0, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETCC(0, 0, 0, 2);
        TTI_SFPLOADI(2, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 2, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 0);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 2);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 2);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 2);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 4);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 4);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 4);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 6);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 6);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 6);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 8);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 8);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 8);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 0xa);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 0xa);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 0xa);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 0xc);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 0xc);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 0xc);
        TTI_SFPLOADI(3, 8, 0x3fb8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(0, 0, 0x4306);
        TTI_SFPLOAD(1, 0, 7, 0xe);
        TTI_SFPMAD(1, 3, 0, 2, 0);
        TTI_SFPLOADI(4, 0, 0x4314);
        TTI_SFPSWAP(0, 9, 2, 9);
        TTI_SFPSWAP(0, 4, 2, 1);
        TTI_SFPEXEXP(0, 2, 4, 0);
        TTI_SFPEXMAN(0, 2, 2, 0);
        TTI_SFPSHFT(0, 4, 2, 0);
        TTI_SFPEXMAN(0, 2, 2, 1);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPLOADI(4, 8, 0x33a8);
        TTI_SFPCAST(2, 2, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(4, 0xa, 0x5ade);
        TTI_SFPMAD(3, 2, 4, 3, 0);
        TTI_SFPLOADI(0, 8, 0x3f80);
        TTI_SFPMAD(3, 2, 6, 2, 0);
        TTI_SFPLOADI(3, 0, 0x4380);
        TTI_SFPSETEXP(0, 2, 4, 2);
        TTI_SFPMUL(4, 4, 9, 2, 0);
        TTI_SFPLOADI(0, 0xa, 0x4189);
        TTI_SFPMAD(3, 4, 2, 2, 0);
        TTI_SFPMAD(0, 4, 3, 0, 0);
        TTI_SFPADDI(0x4700, 2, 0);
        TTI_SFPMUL(1, 4, 9, 3, 0);
        TTI_SFPARECIP(0, 2, 4, 0);
        TTI_SFPMAD(2, 4, 12, 2, 2);
        TTI_SFPSETCC(0, 2, 0, 0);
        TTI_SFPMAD(2, 4, 9, 4, 3);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPMUL(0, 4, 9, 0, 0);
        TTI_SFPMAD(7, 4, 10, 2, 0);
        TTI_SFPMUL(3, 0, 9, 3, 0);
        TTI_SFPMUL(1, 2, 9, 0, 0);
        TTI_SFPSETCC(0, 1, 0, 0);
        TTI_SFPMOV(0, 3, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 0xe);
        TTI_SFPXOR(0, 5, 1, 0);
        TTI_SFPAND(5, 1, 2, 1);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 1, 0, 2);
        TTI_SFPLOADI(0, 0, 0x7f80);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 0xe);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_mish_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_mish_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::mish::calculate_exponent_alu_product<ckernel::sfpu::MishBf16Config, ITERATIONS>();
    }
}
inline void init_mish_bf16() { ckernel::sfpu::bf16::mish::init_exponent_alu_product<ckernel::sfpu::MishBf16Config>(); }

}  // namespace ckernel::sfpu
