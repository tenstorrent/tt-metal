// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct HardshrinkBf16Config {
    static constexpr uint32_t kKind = 0x00000001u;
    static constexpr uint32_t kThresholdBits = 0x3f000000u;
    static constexpr uint32_t kComparatorBf16 = 0x00003f01u;
    static constexpr uint32_t kSlopeBits = 0x00000000u;
    static constexpr uint32_t kInterceptBits = 0x00000000u;
    static constexpr uint32_t kRawEqual = 0x00000000u;
    static constexpr uint32_t kBodySlots = 0x00000010u;
    static constexpr uint32_t kRowsPerReplay = 0x00000002u;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_simple_forward.h"
#ifndef TT_POLY_LLK_SIMPLE_FORWARD_SELECTED_V1
#error "selected simple forward runtime required"
#endif
