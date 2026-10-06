// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_simple_algebraic.h"
}
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_simple_forward() {
    // Only the replay's advance mode needs programming. A threshold pair that
    // folds its identity store (always on WH, at 11 slots on BH) advances with
    // INCRWC and never reads it.
    constexpr bool folds = Config::kKind == 1 && Config::kBodySlots == 11;
    if constexpr (Config::kRowsPerReplay && !folds) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
}
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y;
        y = sfpi::simple_threshold_softshift<true>(x, __builtin_bit_cast(float, Config::kThresholdBits));
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu::bf16
