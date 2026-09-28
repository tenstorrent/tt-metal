// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct SoftsignBf16Config {
    static constexpr uint32_t kBodySlots = 19u;
    static constexpr uint32_t kBoundBits = 0x44000000u;
    static constexpr uint32_t kBoundExponent = 0u;
    static constexpr uint32_t kLutSlopes = 0x00000000u;
    static constexpr uint32_t kLutIntercepts = 0x00000000u;
    static constexpr bool kLateRound = false;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_denominator.h"
#ifndef TT_POLY_LLK_ABS_DENOMINATOR_REPLAY_V1
#error "typed abs-denominator replay runtime required"
#endif
#define TT_POLY_SOFTSIGN_BF16_AVAILABLE 1
