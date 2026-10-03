// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_exp2_paired.h"
namespace ckernel::sfpu::bf16 {
constexpr uint32_t kExpHold = ADDR_MOD_3, kExpAdvance = ADDR_MOD_2;

template <typename Config>
inline void init_exp2() {
    static_assert(Config::kDegree == 2u || Config::kDegree == 3u);
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kMultiplierBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree - 1]);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

template <typename Config>
inline void exp2_clamp_pins() {
    // Clamp operands are destroyed by SWAP; SFPLOADI must execute per replay.
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
    if constexpr (Config::kNegativeNanTerminal) {
        // +Inf for exp2_nan_to_upper; the body's SFPEXEXP overwrites L1.
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    }
}

// WH SFPMAD keeps a NaN's sign, and the lower clamp ranks -NaN below 0. As an
// integer, +Inf - y maps a negative NaN to a positive NaN, which the upper clamp
// saturates, and every other negative y to a negative value.
inline void exp2_nan_to_upper() {
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(0, 0, 0, 0);
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == 2 * (10 + Config::kDegree) + 4);
    static_assert(Config::kDegree != 3 || Config::kScaledCoefficientBits[0] == 0x3f800000u);
    constexpr uint32_t replay_slots = Config::kBodySlots;
    constexpr uint32_t last = Config::kScaledCoefficientBits[Config::kDegree == 3 ? 1 : 0];
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, last >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, last & 0xffffu);
    exp2_clamp_pins<Config>();
    // The NaN fix is issued between two recorded parts: the whole body would
    // not fit the 32-slot replay buffer.
    constexpr uint32_t scale_slots = 4;
    if constexpr (Config::kNegativeNanTerminal) {
        TTI_REPLAY(0, scale_slots, 1, 1);
        sfpi::exp2_paired_scale<kExpHold>();
        exp2_nan_to_upper();
        TTI_REPLAY(scale_slots, replay_slots - scale_slots, 1, 1);
        sfpi::exp2_paired_reduce<Config::kDegree, true>();
    } else {
        TTI_REPLAY(0, replay_slots, 1, 1);
        sfpi::exp2_paired_body<Config::kDegree, true, kExpHold>();
    }
    // The complete primitive already owns the target special classes.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kExpHold, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kExpAdvance, 2);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        exp2_clamp_pins<Config>();
        if constexpr (Config::kNegativeNanTerminal) {
            TTI_REPLAY(0, scale_slots, 0, 0);
            exp2_nan_to_upper();
            TTI_REPLAY(scale_slots, replay_slots - scale_slots, 0, 0);
        } else {
            TTI_REPLAY(0, replay_slots, 0, 0);
        }
    }
}
}  // namespace ckernel::sfpu::bf16
