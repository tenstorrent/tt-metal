// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct LeakyReluBf16Config {
    static constexpr uint32_t kSlopeBits = 0x3c23d70au;
    static constexpr uint32_t kBoundaryCount = 0u;
    static constexpr uint32_t kBoundaryRaw = 0x0000u;
    static constexpr uint32_t kBoundaryOutput = 0x0000u;
    static constexpr uint32_t kBodySlots = 7u;
    static constexpr bool kSignedNanIngress = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_slope_max.h"
#ifndef TT_POLY_LLK_SLOPE_MAX_REPLAY_V1
#error "typed unit-high slope replay runtime required"
#endif
#define TT_POLY_LEAKY_RELU_BF16_AVAILABLE 1
