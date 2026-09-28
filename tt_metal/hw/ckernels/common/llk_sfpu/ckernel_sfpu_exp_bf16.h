// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Configuration for the selected paired exponent-ALU total form.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct ExpBf16Config {
    static constexpr uint32_t kDegree = 3u;
    static constexpr uint32_t kScaledCoefficientBits[] = {0x3f800000u, 0x33b22220u, 0x27677e58u, 0x1b1fcb14u};
    static constexpr uint32_t kMultiplierBits = 0x3fb8aa3bu;
    static constexpr uint32_t kBodySlots = 30u;
};
}  // namespace ttpoly_generated
#include "ckernel_sfpu_tt_poly_exp2.h"
#ifndef TT_POLY_LLK_EXP2_REPLAY_V1
#error "typed paired exponential replay runtime required"
#endif
