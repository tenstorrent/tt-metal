// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_horner.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_normalized_log_odds_core.h"
}
namespace ckernel::sfpu::ttpoly {
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
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_NORMALIZED_LOG_ODDS_SELECTED_V1 1
