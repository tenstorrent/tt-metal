// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_softsign_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_softsign_bf16_abs_denominator_replay.h"
namespace ckernel::sfpu::bf16::softsign {
template <typename Config>
inline void init_abs_denominator() {
    ttpoly::softsign::abs_denominator_lut_init<Config::kLutSlopes, Config::kLutIntercepts>();
}

template <typename Config, int Iterations = 32>
inline void calculate_abs_denominator() {
    static_assert(Iterations == 32, "selected compact traversal owns one whole tile");
    constexpr bool raw = Config::kRawNegativeInfinity;
    static_assert(Config::kBodySlots == (raw ? 16 : 14));
    // The suffix is recorded with the core, so each row is one replay.
    constexpr uint32_t slots =
        Config::kBodySlots + ttpoly::softsign::abs_denominator_wh_suffix_slots<Config::kLateRound, raw>();
    static_assert(slots <= 32);
    // Where the row does not fit, the core alone is recorded if it fits.
    auto row_head = [&] {
        ttpoly::softsign::abs_denominator_wh_core<Config::kBoundExponent, Config::kLateRound, ADDR_MOD_3, raw>();
    };
    auto row_tail = [&] { ttpoly::softsign::abs_denominator_wh_suffix<Config::kLateRound, ADDR_MOD_3, raw>(); };
    record_split<0, slots, Config::kBodySlots>(row_head, row_tail);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        replay_split<0, slots, Config::kBodySlots>(row_head, row_tail);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16::softsign
