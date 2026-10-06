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
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * Config::kRowsPerReplay}}.set(
            ADDR_MOD_6);
    }
}
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
    ::ckernel::sfpu::bf16_sfpi::sfploadi(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 65535);
    ::ckernel::sfpu::bf16_sfpi::replay(0, Config::kBodySlots, 1, 1);
    sfpi::simple_gated_pair<false, (Config::kRawEqual != 0u), Config::kRawEqual, 0, 0, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay) {
        ::ckernel::sfpu::bf16_sfpi::replay(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
