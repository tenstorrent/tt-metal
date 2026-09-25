// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_exp2_paired.h"
namespace ckernel::sfpu::ttpoly {
#if defined(ARCH_WORMHOLE)
constexpr uint32_t kExpHold = ADDR_MOD_3, kExpAdvance = ADDR_MOD_2;
#elif defined(ARCH_BLACKHOLE)
constexpr uint32_t kExpHold = ADDR_MOD_7, kExpAdvance = ADDR_MOD_6;
#else
#error "paired exponential requires BH or WH"
#endif

template <typename Config>
inline void init_exp2() {
    static_assert(Config::kDegree == 2u || Config::kDegree == 3u);
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kMultiplierBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree - 1]);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

inline void exp2_clamp_pins() {
    // Clamp operands are destroyed by SWAP; SFPLOADI must execute per replay.
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
}

template <typename Config>
inline void exp2_terminal_and_store_pair() {
    static_assert(Config::kRawNegativeNanInfinity);
    // Original rows still occupy DST. L1/L4 and L2/L6 are dead after rounding;
    // retain the numerical results in L0/L3 and the next-pair pins in L5/L7.
    TTI_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, kExpHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_UINT16, kExpHold, 2);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
    TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
    TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG4, 0);
    TTI_SFPAND(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, 1);
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPAND(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG6, 1);
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kExpHold, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kExpAdvance, 2);
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == 2 * (10 + Config::kDegree) + 4);
    static_assert(Config::kDegree != 3 || Config::kScaledCoefficientBits[0] == 0x3f800000u);
#if defined(ARCH_BLACKHOLE)
    static_assert(Config::kRawNegativeNanInfinity, "BH replay requires its encoded raw terminal");
#else
    static_assert(!Config::kRawNegativeNanInfinity, "WH signed ingress owns the native terminal");
#endif
    constexpr uint32_t replay_slots = Config::kBodySlots - (Config::kRawNegativeNanInfinity ? 2u : 0u);
    constexpr uint32_t last = Config::kScaledCoefficientBits[Config::kDegree == 3 ? 1 : 0];
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, last >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, last & 0xffffu);
    exp2_clamp_pins();
    TTI_REPLAY(0, replay_slots, 1, 1);
    sfpi::exp2_paired_body<Config::kDegree, true, kExpHold>();
    if constexpr (Config::kRawNegativeNanInfinity) {
        exp2_terminal_and_store_pair<Config>();
    } else {
        TTI_SFPSTORE(p_sfpu::LREG0, 0, kExpHold, 0);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, kExpAdvance, 2);
    }
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        exp2_clamp_pins();
        TTI_REPLAY(0, replay_slots, 0, 0);
        if constexpr (Config::kRawNegativeNanInfinity) {
            exp2_terminal_and_store_pair<Config>();
        }
    }
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_EXP2_REPLAY_V1 1
