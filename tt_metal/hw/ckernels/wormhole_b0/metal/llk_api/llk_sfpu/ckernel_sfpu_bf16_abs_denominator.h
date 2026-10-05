// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_bf16_abs_denominator_replay.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_abs_denominator() {
    sfpi::abs_denominator_lut_init<Config::kLutSlopes, Config::kLutIntercepts>();
}

template <typename Config, int Iterations = 32>
inline void calculate_abs_denominator() {
    static_assert(Iterations == 32, "selected compact traversal owns one whole tile");
    constexpr bool raw = Config::kRawNegativeInfinity;
    static_assert(Config::kBodySlots == (raw ? (Config::kLateRound ? 16 : 18) : 14));
    // The suffix is recorded with the core, so each row is one replay.
    constexpr uint32_t slots = Config::kBodySlots + sfpi::abs_denominator_wh_suffix_slots<Config::kLateRound, raw>();
    static_assert(slots <= 32);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    TTI_REPLAY(0, slots, 1, 1);
    sfpi::abs_denominator_wh_core<Config::kBoundExponent, Config::kLateRound, ADDR_MOD_3, raw>();
    sfpi::abs_denominator_wh_suffix<Config::kLateRound, ADDR_MOD_2, raw>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, slots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
