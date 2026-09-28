// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly {
template <typename Config>
constexpr bool slope_max_contract() {
#if defined(ARCH_BLACKHOLE)
    constexpr bool target = !Config::kSignedNanIngress && Config::kBodySlots == (Config::kBoundaryCount ? 22u : 12u);
#elif defined(ARCH_WORMHOLE)
    constexpr bool target = Config::kSignedNanIngress && Config::kBodySlots == (Config::kBoundaryCount ? 13u : 7u);
#else
    constexpr bool target = false;
#endif
    return target && Config::kBoundaryCount <= 1u && Config::kSlopeBits > 0u && Config::kSlopeBits < 0x3f800000u &&
           Config::kBoundaryRaw <= 0xffffu && Config::kBoundaryOutput <= 0xffffu;
}

template <typename Config>
inline void init_slope_max() {
    static_assert(slope_max_contract<Config>());
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kSlopeBits);
#if defined(ARCH_BLACKHOLE)
    if constexpr (Config::kBoundaryCount) {
        sfpi::vConstIntPrgm1 = Config::kBoundaryRaw;
    }
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
#else
    if constexpr (Config::kBoundaryCount) {
        sfpi::vConstIntPrgm2 = Config::kBoundaryRaw;
    }
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#endif
}

template <typename Config, int Iterations = 8>
inline void calculate_slope_max() {
    static_assert(slope_max_contract<Config>());
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    // Preserve the selected S55 replay body. The upstream callback owns face
    // traversal and WH addrmod-base lifetime; no deployment tile reset belongs here.
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
#if defined(ARCH_BLACKHOLE)
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
    TTI_SFPMUL(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 0);
    TTI_SFPMUL(p_sfpu::LREG12, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    if constexpr (Config::kBoundaryCount) {
        TTI_SFPLOAD(p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_UINT16, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_UINT16, ADDR_MOD_7, 2);
        TTI_SFPXOR(0, p_sfpu::LREG13, p_sfpu::LREG2, 0);
        TTI_SFPXOR(0, p_sfpu::LREG13, p_sfpu::LREG3, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::kBoundaryOutput);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, Config::kBoundaryOutput);
        TTI_SFPENCC(0, 0, 0, 0);
    }
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG1,
        p_sfpu::LREG1,
        p_sfpu::LREG1,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_6, 2);
#if defined(SCALAR_ROW_UNROLL_DISABLE)
#pragma GCC unroll 4
#else
#pragma GCC unroll 16
#endif
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
#else
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    if constexpr (Config::kBoundaryCount) {
        TTI_SFPLOAD(p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, ADDR_MOD_3, 0);
    }
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
    TTI_SFPMUL(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    if constexpr (Config::kBoundaryCount) {
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);
        TTI_SFPXOR(0, p_sfpu::LREG14, p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, Config::kBoundaryOutput);
        TTI_SFPENCC(0, 0, 0, 0);
    }
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_2, 0);
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_SLOPE_MAX_REPLAY_V1 1
