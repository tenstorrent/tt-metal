// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct SoftsignBf16Config {
    static constexpr uint32_t kBodySlots = 16u;
    static constexpr uint32_t kBoundBits = 0x00000000u;
    static constexpr uint32_t kBoundExponent = 136u;
    static constexpr uint32_t kLutSlopes = 0xb549b9a8u;
    static constexpr uint32_t kLutIntercepts = 0x3c9f3ed4u;
    static constexpr bool kLateRound = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_denominator.h"
#ifndef TT_POLY_LLK_ABS_DENOMINATOR_REPLAY_V1
#error "typed abs-denominator replay runtime required"
#endif
#define TT_POLY_SOFTSIGN_BF16_AVAILABLE 1
