// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_reciprocal_bf16_replay_window.h"

namespace ckernel::sfpu::bf16::reciprocal {
// Two DEST rows per replay; the seed c0..c2 is whatever stock's reciprocal init left in Prgm0..2, and either seed
// rounds correctly for every BF16 mantissa. The scale carries the sign, so no compare is needed for 0, Inf or NaN.
constexpr std::uint32_t kNewtonReciprocalSlots = 26u;

inline void newton_reciprocal_pair_head() {
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, 0);       // x (row A)
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, 2);       // x (row B)
    TTI_SFPSETMAN(0, p_sfpu::LREG3, p_sfpu::LREG2, 1);  // sign and exponent of x
    TTI_SFPSETMAN(0, p_sfpu::LREG6, p_sfpu::LREG5, 1);
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0);  // n = -m
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG0, 0);  // c0 n + c1
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG6, p_sfpu::LREG13, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG2, 6);                             // 0x7F800000 - bits
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG0, 0);  // y = (..) n + c2
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG14, p_sfpu::LREG4, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, 6);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, 0);  // t = n y + 1
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG2, 0);                                      // s = sign(x) 2^(254 - e)
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // u = t t + t
}

inline void newton_reciprocal_pair_tail() {
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPMULI(0x3f00, p_sfpu::LREG5, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // y = y u + y
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    // Round y before the scale: s is a power of two, so y s is exact. FP32 -> BF16 reads no VB, so VB is L0.
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG4,
        p_sfpu::LREG4,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // y s
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
    // ADDR_MOD_3 (ADDR_MOD_7 past the SFPU's +4 base) holds DEST at zero step in every SFPU init;
    // ADDR_MOD_6 belongs to the FPU ops between calls, so an INCRWC after each pair steps DEST.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_3, 2);
}

template <typename Config, int Iterations = 8>
inline void calculate_newton_reciprocal() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == kNewtonReciprocalSlots);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80);  // +Inf bits
    auto row_head = [&] { newton_reciprocal_pair_head(); };
    auto row_tail = [&] { newton_reciprocal_pair_tail(); };
    record_split<0, Config::kBodySlots, 16>(row_head, row_tail);
    TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 8
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        replay_split<0, Config::kBodySlots, 16>(row_head, row_tail);
        TTI_INCRWC(0, 4, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16::reciprocal
