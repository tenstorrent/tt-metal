// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct Log10Bf16Config {
    static constexpr uint32_t kDegree = 6u;
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3fb8aa3bu, 0xbf38373bu, 0x3eeca3edu, 0xbe918bdbu, 0x3e00896cu, 0xbcd973dau};
    static constexpr uint32_t kScaleBits = 0x3e9a209bu;
    static constexpr uint32_t kBodySlots = 32u;
    static constexpr uint32_t kTailSlots = 1u;
    static constexpr uint32_t kTerminalSlots = 8u;
    static constexpr uint32_t kInputMinNormalBits = 0x00800000u;
    static constexpr bool kRawPartition = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_log2.h"
#ifndef TT_POLY_LLK_LOG2_REPLAY_V1
#error "typed logarithm replay runtime required"
#endif
#define TT_POLY_LOG10_BF16_AVAILABLE 1
