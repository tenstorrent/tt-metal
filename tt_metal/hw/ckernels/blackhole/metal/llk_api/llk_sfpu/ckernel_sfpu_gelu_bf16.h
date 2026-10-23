// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct GeluBf16Config {
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f27dc51u),
        __builtin_bit_cast(float, 0x3eaa0299u)};
    static constexpr unsigned kExpDegree = 2u;
    static constexpr float kCoreCoefficients[] = {
        __builtin_bit_cast(float, 0x3ecb4b84u),
        __builtin_bit_cast(float, 0xbd8262c6u),
        __builtin_bit_cast(float, 0x3c095ac4u),
        __builtin_bit_cast(float, 0xba3e17aau),
        __builtin_bit_cast(float, 0x3818921cu),
        __builtin_bit_cast(float, 0xb5556e76u)};
    static constexpr unsigned kCoreDegree = 5u;
    static constexpr float kDecayCoefficients[] = {
        __builtin_bit_cast(float, 0xbe9f9763u),
        __builtin_bit_cast(float, 0x3cd3682eu),
        __builtin_bit_cast(float, 0x3b2ecf9fu),
        __builtin_bit_cast(float, 0x38c1d21bu)};
    static constexpr unsigned kDecayDegree = 3u;
    static constexpr float kNegativeTerminal = __builtin_bit_cast(float, 0xc1530000u);
    static constexpr float kDecayCoreBoundary = __builtin_bit_cast(float, 0xc0480000u);
    static constexpr float kPositiveIdentity = __builtin_bit_cast(float, 0x40310000u);
    static constexpr float kOriginSlope = __builtin_bit_cast(float, 0x3f000000u);
    static constexpr unsigned kBodySlots = 25u;
    static constexpr unsigned kReplayCapacity = 32u;
    static constexpr unsigned kPeakLive = 8u;
    static constexpr unsigned kOriginMask = 32767u;
    static constexpr unsigned kOriginValue = 32513u;
    static constexpr unsigned kOriginMantissa = 0u;
    static constexpr unsigned kOriginPatterns = 2u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_gelu_bf16_zone_affine_even_decay.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_gelu() {
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
// ckernel::sfpu::bf16::gelu::calculate_zone_affine_even_decay<ckernel::sfpu::GeluBf16Config, 32> in a stall-filling
// order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every BF16 input;
// other ITERATIONS run the compiled body itself.
inline void calculate_gelu_bf16_scheduled() {
    TTI_SFPLOADI(7, 2, 0xaa3b);
    TTI_SFPLOADI(7, 8, 0xbf38);
    TTI_SFPLOADI(6, 2, 0x299);
    TTI_SFPLOADI(5, 2, 0xdc51);
    TTI_SFPLOADI(6, 8, 0x27aa);
    TTI_SFPLOADI(5, 8, 0x33a7);
// A loop, not unrolled: the body stays in the instruction cache.
#pragma GCC unroll 1
    for (int pass = 0; pass < 8; ++pass) {
        TTI_SFPLOAD(2, 0, 7, 0);
        TTI_SFPMUL(2, 2, 9, 3, 0);
        TTI_SFPLOADI(0, 0, 0x42fe);
        TTI_SFPMAD(3, 7, 0, 1, 0);
        TTI_SFPLOADI(0, 2, 0xd21b);
        TTI_SFPEXEXP(0, 1, 3, 0);
        TTI_SFPEXMAN(0, 1, 1, 0);
        TTI_SFPLOADI(4, 2, 0xcf9f);
        TTI_SFPSHFT(0, 3, 1, 0);
        TTI_SFPLOADI(0, 8, 0x38c1);
        TTI_SFPLOADI(4, 8, 0x3b2e);
        TTI_SFPEXMAN(0, 1, 3, 1);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x682e);
        TTI_SFPCAST(3, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3cd3);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x9763);
        TTI_SFPLOADI(4, 8, 0xbe9f);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPMAD(6, 3, 5, 4, 0);
        TTI_SFPEXEXP(0, 1, 1, 1);
        TTI_SFPMAD(4, 3, 10, 3, 0);
        TTI_SFPSHFT(0x17, 1, 1, 7);
        TTI_SFPMUL(3, 0, 9, 0, 0);
        TTI_SFPLOADI(3, 0, 0xc048);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPGT(0, 3, 2, 1);
        TTI_SFPLOADI(3, 2, 0x6e76);
        TTI_SFPLOADI(4, 2, 0x921c);
        TTI_SFPLOADI(3, 8, 0xb555);
        TTI_SFPLOADI(4, 8, 0x3818);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x17aa);
        TTI_SFPLOADI(4, 8, 0xba3e);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x5ac4);
        TTI_SFPLOADI(4, 8, 0x3c09);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x62c6);
        TTI_SFPLOADI(4, 8, 0xbd82);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x4b84);
        TTI_SFPLOADI(4, 8, 0x3ecb);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPLOADI(3, 0, 0x3f00);
        TTI_SFPMAD(2, 1, 3, 1, 0);
        TTI_SFPMUL(2, 1, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc153);
        TTI_SFPLE(0, 1, 2, 1);
        TTI_SFPMUL(2, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0x4031);
        TTI_SFPGT(0, 1, 2, 1);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 0);
        TTI_SFPLOADI(3, 2, 0x7f01);
        TTI_SFPXOR(0, 3, 1, 0);
        TTI_SFPLOADI(3, 2, 0x7fff);
        TTI_SFPAND(1, 3, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETMAN(0, 2, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 0);
        TTI_SFPLOAD(2, 0, 7, 2);
        TTI_SFPMUL(2, 2, 9, 3, 0);
        TTI_SFPLOADI(0, 0, 0x42fe);
        TTI_SFPMAD(3, 7, 0, 1, 0);
        TTI_SFPLOADI(0, 2, 0xd21b);
        TTI_SFPEXEXP(0, 1, 3, 0);
        TTI_SFPEXMAN(0, 1, 1, 0);
        TTI_SFPLOADI(4, 2, 0xcf9f);
        TTI_SFPSHFT(0, 3, 1, 0);
        TTI_SFPLOADI(0, 8, 0x38c1);
        TTI_SFPLOADI(4, 8, 0x3b2e);
        TTI_SFPEXMAN(0, 1, 3, 1);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x682e);
        TTI_SFPCAST(3, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3cd3);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x9763);
        TTI_SFPLOADI(4, 8, 0xbe9f);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPMAD(6, 3, 5, 4, 0);
        TTI_SFPEXEXP(0, 1, 1, 1);
        TTI_SFPMAD(4, 3, 10, 3, 0);
        TTI_SFPSHFT(0x17, 1, 1, 7);
        TTI_SFPMUL(3, 0, 9, 0, 0);
        TTI_SFPLOADI(3, 0, 0xc048);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPGT(0, 3, 2, 1);
        TTI_SFPLOADI(3, 2, 0x6e76);
        TTI_SFPLOADI(4, 2, 0x921c);
        TTI_SFPLOADI(3, 8, 0xb555);
        TTI_SFPLOADI(4, 8, 0x3818);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x17aa);
        TTI_SFPLOADI(4, 8, 0xba3e);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x5ac4);
        TTI_SFPLOADI(4, 8, 0x3c09);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x62c6);
        TTI_SFPLOADI(4, 8, 0xbd82);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x4b84);
        TTI_SFPLOADI(4, 8, 0x3ecb);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPLOADI(3, 0, 0x3f00);
        TTI_SFPMAD(2, 1, 3, 1, 0);
        TTI_SFPMUL(2, 1, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc153);
        TTI_SFPLE(0, 1, 2, 1);
        TTI_SFPMUL(2, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0x4031);
        TTI_SFPGT(0, 1, 2, 1);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 2);
        TTI_SFPLOADI(3, 2, 0x7f01);
        TTI_SFPXOR(0, 3, 1, 0);
        TTI_SFPLOADI(3, 2, 0x7fff);
        TTI_SFPAND(1, 3, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETMAN(0, 2, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 2);
        TTI_SFPLOAD(2, 0, 7, 4);
        TTI_SFPMUL(2, 2, 9, 3, 0);
        TTI_SFPLOADI(0, 0, 0x42fe);
        TTI_SFPMAD(3, 7, 0, 1, 0);
        TTI_SFPLOADI(0, 2, 0xd21b);
        TTI_SFPEXEXP(0, 1, 3, 0);
        TTI_SFPEXMAN(0, 1, 1, 0);
        TTI_SFPLOADI(4, 2, 0xcf9f);
        TTI_SFPSHFT(0, 3, 1, 0);
        TTI_SFPLOADI(0, 8, 0x38c1);
        TTI_SFPLOADI(4, 8, 0x3b2e);
        TTI_SFPEXMAN(0, 1, 3, 1);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x682e);
        TTI_SFPCAST(3, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3cd3);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x9763);
        TTI_SFPLOADI(4, 8, 0xbe9f);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPMAD(6, 3, 5, 4, 0);
        TTI_SFPEXEXP(0, 1, 1, 1);
        TTI_SFPMAD(4, 3, 10, 3, 0);
        TTI_SFPSHFT(0x17, 1, 1, 7);
        TTI_SFPMUL(3, 0, 9, 0, 0);
        TTI_SFPLOADI(3, 0, 0xc048);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPGT(0, 3, 2, 1);
        TTI_SFPLOADI(3, 2, 0x6e76);
        TTI_SFPLOADI(4, 2, 0x921c);
        TTI_SFPLOADI(3, 8, 0xb555);
        TTI_SFPLOADI(4, 8, 0x3818);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x17aa);
        TTI_SFPLOADI(4, 8, 0xba3e);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x5ac4);
        TTI_SFPLOADI(4, 8, 0x3c09);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x62c6);
        TTI_SFPLOADI(4, 8, 0xbd82);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x4b84);
        TTI_SFPLOADI(4, 8, 0x3ecb);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPLOADI(3, 0, 0x3f00);
        TTI_SFPMAD(2, 1, 3, 1, 0);
        TTI_SFPMUL(2, 1, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc153);
        TTI_SFPLE(0, 1, 2, 1);
        TTI_SFPMUL(2, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0x4031);
        TTI_SFPGT(0, 1, 2, 1);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 4);
        TTI_SFPLOADI(3, 2, 0x7f01);
        TTI_SFPXOR(0, 3, 1, 0);
        TTI_SFPLOADI(3, 2, 0x7fff);
        TTI_SFPAND(1, 3, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETMAN(0, 2, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 4);
        TTI_SFPLOAD(2, 0, 7, 6);
        TTI_SFPMUL(2, 2, 9, 3, 0);
        TTI_SFPLOADI(0, 0, 0x42fe);
        TTI_SFPMAD(3, 7, 0, 1, 0);
        TTI_SFPLOADI(0, 2, 0xd21b);
        TTI_SFPEXEXP(0, 1, 3, 0);
        TTI_SFPEXMAN(0, 1, 1, 0);
        TTI_SFPLOADI(4, 2, 0xcf9f);
        TTI_SFPSHFT(0, 3, 1, 0);
        TTI_SFPLOADI(0, 8, 0x38c1);
        TTI_SFPLOADI(4, 8, 0x3b2e);
        TTI_SFPEXMAN(0, 1, 3, 1);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x682e);
        TTI_SFPCAST(3, 3, 0);
        TTI_SFPLOADI(4, 8, 0x3cd3);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPLOADI(4, 2, 0x9763);
        TTI_SFPLOADI(4, 8, 0xbe9f);
        TTI_SFPMAD(0, 2, 4, 0, 0);
        TTI_SFPMAD(6, 3, 5, 4, 0);
        TTI_SFPEXEXP(0, 1, 1, 1);
        TTI_SFPMAD(4, 3, 10, 3, 0);
        TTI_SFPSHFT(0x17, 1, 1, 7);
        TTI_SFPMUL(3, 0, 9, 0, 0);
        TTI_SFPLOADI(3, 0, 0xc048);
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_SFPMUL(2, 2, 9, 1, 0);
        TTI_SFPGT(0, 3, 2, 1);
        TTI_SFPLOADI(3, 2, 0x6e76);
        TTI_SFPLOADI(4, 2, 0x921c);
        TTI_SFPLOADI(3, 8, 0xb555);
        TTI_SFPLOADI(4, 8, 0x3818);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x17aa);
        TTI_SFPLOADI(4, 8, 0xba3e);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x5ac4);
        TTI_SFPLOADI(4, 8, 0x3c09);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x62c6);
        TTI_SFPLOADI(4, 8, 0xbd82);
        TTI_SFPMAD(3, 1, 4, 3, 0);
        TTI_SFPLOADI(4, 2, 0x4b84);
        TTI_SFPLOADI(4, 8, 0x3ecb);
        TTI_SFPMAD(3, 1, 4, 1, 0);
        TTI_SFPLOADI(3, 0, 0x3f00);
        TTI_SFPMAD(2, 1, 3, 1, 0);
        TTI_SFPMUL(2, 1, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0xc153);
        TTI_SFPLE(0, 1, 2, 1);
        TTI_SFPMUL(2, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOADI(1, 0, 0x4031);
        TTI_SFPGT(0, 1, 2, 1);
        TTI_SFPMOV(0, 2, 0, 0);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFPLOAD(1, 6, 7, 6);
        TTI_SFPLOADI(3, 2, 0x7f01);
        TTI_SFPXOR(0, 3, 1, 0);
        TTI_SFPLOADI(3, 2, 0x7fff);
        TTI_SFPAND(1, 3, 1, 1);
        TTI_SFPSETCC(0, 1, 0, 6);
        TTI_SFPSETMAN(0, 2, 0, 1);
        TTI_SFPENCC(3, 0, 0, 0xa);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 1);
        TTI_SFPSTORE(0, 0, 7, 6);
        TTI_INCRWC(0, 7, 0, 0);
        TTI_INCRWC(0, 1, 0, 0);
    }
}

template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_gelu_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_gelu_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::gelu::calculate_zone_affine_even_decay<ckernel::sfpu::GeluBf16Config, ITERATIONS>();
    }
}
inline void init_gelu_bf16() {
    ckernel::sfpu::bf16::gelu::init_zone_affine_even_decay<ckernel::sfpu::GeluBf16Config>();
}

}  // namespace ckernel::sfpu
