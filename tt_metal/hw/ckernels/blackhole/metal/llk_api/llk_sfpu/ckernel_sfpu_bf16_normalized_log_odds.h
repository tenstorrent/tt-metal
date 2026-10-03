// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_horner.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_normalized_log_odds_core.h"
}
namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_normalized_log_odds() {
    static_assert(Iterations == 32, "normalized log odds requires a complete tile");
    // Retain the selected canonical polynomial preamble. The surrounding
    // upstream caller still owns SFPU start/done and destination counters.
    sfpi::vConstFloatPrgm1 = Config::kLut[6];
    sfpi::vConstFloatPrgm2 = Config::kLut[5];
    sfpi::vConstFloatPrgm0 = Config::kLut[4];
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ckernel::ADDR_MOD_6);
    sfpi::normalized_log_odds_tile<Config>(Config::kLut, [](sfpi::vFloat x) { return x; });
}
}  // namespace ckernel::sfpu::bf16
