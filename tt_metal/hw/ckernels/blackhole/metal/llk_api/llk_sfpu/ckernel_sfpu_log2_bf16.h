// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct Log2Bf16Config {
    static constexpr uint32_t kDegree = 6u;
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3fb8a9d6u, 0xbf386deeu, 0x3ef03878u, 0xbe9b298cu, 0x3e158d8cu, 0xbd0d1240u};
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 30u;
    static constexpr uint32_t kTerminalSlots = 7u;
    static constexpr uint32_t kInputMinNormalBits = 0x00800000u;
    static constexpr bool kRawPartition = false;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_log2.h"
#ifndef TT_POLY_LLK_LOG2_REPLAY_V1
#error "typed logarithm replay runtime required"
#endif
