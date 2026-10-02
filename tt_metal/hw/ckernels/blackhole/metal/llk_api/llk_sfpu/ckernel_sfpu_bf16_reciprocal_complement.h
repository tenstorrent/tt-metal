// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
}
#include "ckernel_sfpu_bf16_reciprocal_complement_core.h"
namespace ckernel::sfpu::bf16 {
// Replay slots of one BH row: 12 ahead of the Horner rungs, two per rung (a gap slot,
// then the MAD), one more where degree seven reloads LREG3, and the 4-slot tail.
constexpr unsigned reciprocal_complement_body_slots(unsigned degree) {
    return degree == 8 ? 32u : 12u + 2u * (degree - 1u) + (degree == 7 ? 1u : 0u) + 4u;
}
template <class Config>
inline void init_reciprocal_complement() {
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_reciprocal_complement() {
    static_assert(Iterations == 32);
    static_assert(Config::kDegree >= 4 && Config::kDegree <= 8);
    static_assert(Config::kBodySlots == reciprocal_complement_body_slots(Config::kDegree));
    static_assert(Config::kShadowBase == 96 && Config::kShadowRows == 32);
    static_assert((Config::kShadowBase + Config::kShadowRows) * 2 <= DEST_REGISTER_HALF_SIZE);
    sfpi::vConstFloatPrgm1 = Config::kCoefficients[Config::kDegree];
    sfpi::vConstFloatPrgm2 = Config::kCoefficients[Config::kDegree - 1];
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    TTI_SFPSETSGN(0, p_sfpu::LREG2, 15, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, Config::kMacroSequenceBits >> 16);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, Config::kMacroSequenceBits & 0xffffu);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
#pragma GCC unroll 8
    for (int d = 0; d < 32; ++d) {
        sfpi::vUInt raw = sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>();
        sfpi::dst_reg[Config::kShadowBase + d].template mode<sfpi::DataLayout::U16>() = raw;
    }
    // Degree eight keeps c6 for LREG3; below it the rungs start at c[kDegree - 2].
    constexpr unsigned kTop = Config::kDegree == 8 ? 5u : Config::kDegree - 2u;
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[kTop] >> 16);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[kTop] & 0xffffu);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[kTop - 1] >> 16);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[kTop - 1] & 0xffffu);
    if constexpr (kTop >= 3) {
        TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[kTop - 2] >> 16);
        TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[kTop - 2] & 0xffffu);
    }
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, Config::kComplementBits >> 16);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, Config::kComplementBits & 0xffffu);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    sfpi::reciprocal_complement_body<Config, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int d = 1; d < 32; ++d) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    // The selected suffix is a second traversal. This internal phase reset
    // must precede its result/shadow loads, independently of caller cleanup.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#pragma GCC unroll 8
    for (int d = 0; d < 32; ++d) {
        sfpi::vFloat result = sfpi::dst_reg[d];
        sfpi::vUInt raw = sfpi::dst_reg[Config::kShadowBase + d].template mode<sfpi::DataLayout::U16>();
        sfpi::nan_union_class_terminal<1>(raw, result);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
