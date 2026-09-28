// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_tt_poly_abs_denominator_replay.h"
namespace ckernel::sfpu::ttpoly {
template <typename Config>
inline void init_abs_denominator() {
#if defined(ARCH_BLACKHOLE)
    sfpu_reciprocal_init<false>();
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kBoundBits);
#elif defined(ARCH_WORMHOLE)
    sfpi::abs_denominator_lut_init<Config::kLutSlopes, Config::kLutIntercepts>();
#else
#error "selected abs denominator replay requires BH or WH"
#endif
}

template <typename Config, int Iterations = 32>
inline void calculate_abs_denominator() {
    static_assert(Iterations == 32, "selected compact traversal owns one whole tile");
#if defined(ARCH_BLACKHOLE)
    static_assert(Config::kBodySlots == 19);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    sfpi::abs_denominator_loadmacro_init();
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    sfpi::abs_denominator_bh_body<ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
#elif defined(ARCH_WORMHOLE)
    static_assert(Config::kBodySlots == (Config::kLateRound ? 16 : 18));
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#pragma GCC unroll 1
    for (int row = 0; row < Iterations; ++row) {
        if (row == 0) {
            TTI_REPLAY(0, Config::kBodySlots, 1, 1);
            sfpi::abs_denominator_wh_core<Config::kBoundExponent, Config::kLateRound, ADDR_MOD_3>();
        } else {
            TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        }
        sfpi::abs_denominator_wh_suffix<Config::kLateRound, ADDR_MOD_2>();
    }
#else
#error "selected abs denominator replay requires BH or WH"
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_ABS_DENOMINATOR_REPLAY_V1 1
