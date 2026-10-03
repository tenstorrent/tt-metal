// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16 {
constexpr uint32_t kAbsHold = ADDR_MOD_3;
constexpr uint32_t kAbsAdvance = ADDR_MOD_2;

template <typename Config>
inline void init_abs_value() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

template <typename Config, int Iterations = 8>
inline void calculate_abs_value() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    // The typed exhaustive proof absorbs the encoded terminal into the native
    // load/FLOAT-SFPABS/store path. The upstream wrapper owns face traversal.
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, Config::kLoadFormat, kAbsHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, Config::kLoadFormat, kAbsHold, 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, Config::kAbsMode);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, Config::kAbsMode);
    TTI_SFPSTORE(p_sfpu::LREG0, Config::kStoreFormat, kAbsHold, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, Config::kStoreFormat, kAbsAdvance, 2);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
