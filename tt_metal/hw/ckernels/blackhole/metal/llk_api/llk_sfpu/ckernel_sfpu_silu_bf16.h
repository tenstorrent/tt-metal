// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct SiluBf16Config {
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kBiasBf16 = 0x430fu;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_silu_bf16_exponent_alu_product.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_silu() {
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
// ckernel::sfpu::bf16::silu::calculate_exponent_alu_product<ckernel::sfpu::SiluBf16Config, 32> in a stall-filling
// order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every BF16 input;
// other ITERATIONS run the compiled body itself.
inline void calculate_silu_bf16_scheduled() {
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 0);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPLOADI(4, 8, 0x3f80);
        TTI_SFPLOADI(4, 0xa, 0x3884);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 0);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 2);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 2);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 4);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 4);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 6);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 6);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 8);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 8);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 0xa);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 0xa);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 0xc);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPDIVP2(0x7e, 2, 0, 1);
        TTI_SFPMAD(6, 5, 7, 5, 1);
        TTI_SFPADDI(0xbfff, 0, 0);
        TTI_SFPSWAP(0, 5, 1, 1);
        TTI_SFPMULI(0x3f00, 5, 0);
        TTI_SFPSETCC(0, 0, 0, 6);
        TTI_SFPSETMAN(0, 1, 5, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 5, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 5, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 0xc);
        TTI_SFPLOADI(3, 8, 0xbfb8);
        TTI_SFPLOAD(1, 0, 7, 0xe);
        TTI_SFPLOADI(3, 0xa, 0xaa3b);
        TTI_SFPLOADI(5, 0, 0x430f);
        TTI_SFPSETSGN(0, 1, 2, 1);
        TTI_SFPMAD(2, 3, 5, 7, 0);
        TTI_SFPLOADI(3, 8, 0x27ac);
        TTI_SFPEXEXP(0, 7, 5, 0);
        TTI_SFPEXMAN(0, 7, 0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPLOADI(6, 8, 0x33a8);
        TTI_SFPCAST(5, 5, 0);
        TTI_SFPLOADI(3, 0xa, 0xa410);
        TTI_SFPLOADI(6, 0xa, 0x5ade);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPGT(0, 9, 7, 8);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPAND(0, 7, 0, 1);
        TTI_SFPSETEXP(0, 5, 0, 2);
        TTI_SFPMUL(2, 0, 9, 6, 0);
        TTI_SFPMULI(0x3780, 0, 0);
        TTI_SFPADD(10, 2, 1, 7, 0);
        TTI_SFPADD(10, 0, 10, 0, 0);
        TTI_SFPMULI(0x3800, 6, 0);
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPDIVP2(0x7e, 2, 3, 1);
        TTI_SFPMAD(0, 5, 9, 5, 3);
        TTI_SFPADDI(0xbfff, 3, 0);
        TTI_SFPMAD(6, 5, 7, 0, 1);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, 0, 1, 1);
        TTI_SFPMULI(0x3f00, 0, 0);
        TTI_SFPSETCC(0, 3, 0, 6);
        TTI_SFPSETMAN(0, 1, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, 2, 0, 1);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 2, 1);
        TTI_SFPSTORE(2, 0, 7, 0xe);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_silu_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_silu_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::silu::calculate_exponent_alu_product<ckernel::sfpu::SiluBf16Config, ITERATIONS>();
    }
}
inline void init_silu_bf16() { ckernel::sfpu::bf16::silu::init_exponent_alu_product<ckernel::sfpu::SiluBf16Config>(); }

}  // namespace ckernel::sfpu
